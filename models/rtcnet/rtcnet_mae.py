from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .rtcnet_modules import Conv, RTCBlock, ConvMLP
except:
    from rtcnet_modules import Conv, RTCBlock, ConvMLP
   

## Real-time Convolutional Backbone
class MAE_RTCBackbone(nn.Module):
    def __init__(self,
                 width=1.0,
                 depth=1.0,
                 ratio=1.0,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False,
                 in_channels: int = 3,
                 mask_patch_size: int = 32,
                 mask_ratio: float = 0.75,
                 is_train: bool = False,
                 norm_pix_loss: bool = False,
                 de_num_layers: int = 12,
                 dropout: float = 0.1,
                 ):
        super(MAE_RTCBackbone, self).__init__()
        # ---------------- Basic parameters ----------------
        self.width_factor = width
        self.depth_factor = depth
        self.last_stage_factor = ratio
        self.feat_dims = [round(64 * width), round(128 * width), round(256 * width), round(512 * width), round(512 * width * ratio)]
        self.is_train = is_train
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        self.norm_pix_loss = norm_pix_loss
        self.out_stride = 32
        # ---------------- Network parameters ----------------
        ## P1/2
        self.layer_1 = Conv(3, self.feat_dims[0], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type)
        ## P2/4
        self.layer_2 = nn.Sequential(
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            RTCBlock(in_dim     = self.feat_dims[1],
                     out_dim    = self.feat_dims[1],
                     num_blocks = round(3*depth),
                     shortcut   = True,
                     act_type   = act_type,
                     norm_type  = norm_type,
                     depthwise  = depthwise)
        )
        ## P3/8
        self.layer_3 = nn.Sequential(
            Conv(self.feat_dims[1], self.feat_dims[2], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            RTCBlock(in_dim     = self.feat_dims[2],
                     out_dim    = self.feat_dims[2],
                     num_blocks = round(6*depth),
                     shortcut   = True,
                     act_type   = act_type,
                     norm_type  = norm_type,
                     depthwise  = depthwise)
        )
        ## P4/16
        self.layer_4 = nn.Sequential(
            Conv(self.feat_dims[2], self.feat_dims[3], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            RTCBlock(in_dim     = self.feat_dims[3],
                     out_dim    = self.feat_dims[3],
                     num_blocks = round(6*depth),
                     shortcut   = True,
                     act_type   = act_type,
                     norm_type  = norm_type,
                     depthwise  = depthwise)
        )
        ## P5/32
        self.layer_5 = nn.Sequential(
            Conv(self.feat_dims[3], self.feat_dims[4], k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            RTCBlock(in_dim     = self.feat_dims[4],
                     out_dim    = self.feat_dims[4],
                     num_blocks = round(3*depth),
                     shortcut   = True,
                     act_type   = act_type,
                     norm_type  = norm_type,
                     depthwise  = depthwise)
        )
        ## Out layer
        self.decoder = nn.Sequential(*[ConvMLP(in_dim     = self.feat_dims[4],
                                               hidden_dim = self.feat_dims[4] * 4,
                                               out_dim    = self.feat_dims[4],
                                               drop       = dropout)
                                               for _ in range(de_num_layers)] + 
                                      [nn.Conv2d(self.feat_dims[4], in_channels * self.out_stride ** 2, kernel_size=1)])

    def patchify(self, imgs, patch_size):
        """
        imgs: (B, 3, H, W)
        x:    (B, N, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x, patch_size):
        """
        x:    (B, N, patch_size**2 *3)
        imgs: (B, 3, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs

    def random_masking(self, x):
        # ----------------- Step-1: Patch embed -----------------
        # Patchify: [B, C, H, W] -> [B, N, C*P*P]
        patches = self.patchify(x, self.mask_patch_size)
        B, N, C = patches.shape

        # ----------------- Step-2: Random masking -----------------
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)        # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore the original position of each patch

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # [B, N_nomask, 3*P*P]
        keep_patches = torch.gather(patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        # unshuffle to get the masked image [B, N, 3*P*P]
        mask_patches = torch.zeros(B, N-len_keep, C, device=x.device)
        x_masked = torch.cat([keep_patches, mask_patches], dim=1)
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = mask.unsqueeze(-1).expand(-1, -1, C)

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))

        # ----------------- Step-3: Reshape masked patches to image format -----------------
        x_masked = self.unpatchify(x_masked, self.mask_patch_size)
        mask = self.unpatchify(mask, self.mask_patch_size)

        return x_masked, mask

    def compute_loss(self, x, output):
        """
        x:    [B, 3, H, W]
        pred: [B, 3, H, W]
        mask: [B, 3, H, W] 0 is keep, 1 is remove, 
        """
        if self.norm_pix_loss:
            mean = x.mean(dim=1, keepdim=True)
            var  = x.var(dim=1, keepdim=True)
            x    = (x - mean) / (var + 1.e-6)**.5
        pred, mask = output["x_pred"], output["mask"]
        loss = (pred - x) ** 2
        loss = loss.mean(1)
        mask = mask[:, 0, :, :]
        loss = (loss * mask).sum() / mask.sum()
        
        return loss

    def forward(self, x):
        imgs = x
        x_masked, mask = self.random_masking(x)

        c1 = self.layer_1(x_masked)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        x = self.decoder(c5)

        # Reshape: [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        # Unpatchify: [B, N, C] -> [B, 3, H, W]
        x = self.unpatchify(x, patch_size=32)

        output = {
            'x_pred': x,   # [B, 3, H, W]
            'mask': mask   # [B, 3, H, W]
        }

        if self.is_train:
            loss = self.compute_loss(imgs, output)
            output["loss"] = loss

        return output


# ------------------------ Model Functions ------------------------
def mae_rtcnet_n(in_channels, mask_patch_size, mask_ratio, is_train=False, norm_pix_loss=False) -> MAE_RTCBackbone:
    return MAE_RTCBackbone(width=0.25,
                           depth=0.34,
                           ratio=2.0,
                           act_type='silu',
                           norm_type='BN',
                           depthwise=False,
                           in_channels=in_channels,
                           mask_patch_size=mask_patch_size,
                           mask_ratio=mask_ratio,
                           is_train=is_train,
                           norm_pix_loss=norm_pix_loss,
                           de_num_layers=8,
                           dropout=0.1
                           )

def mae_rtcnet_t(in_channels, mask_patch_size, mask_ratio, is_train=False, norm_pix_loss=False) -> MAE_RTCBackbone:
    return MAE_RTCBackbone(width=0.375,
                           depth=0.34,
                           ratio=2.0,
                           act_type='silu',
                           norm_type='BN',
                           depthwise=False,
                           in_channels=in_channels,
                           mask_patch_size=mask_patch_size,
                           mask_ratio=mask_ratio,
                           is_train=is_train,
                           norm_pix_loss=norm_pix_loss,
                           de_num_layers=8,
                           dropout=0.1
                           )

def mae_rtcnet_s(in_channels, mask_patch_size, mask_ratio, is_train=False, norm_pix_loss=False) -> MAE_RTCBackbone:
    return MAE_RTCBackbone(width=0.50,
                           depth=0.34,
                           ratio=2.0,
                           act_type='silu',
                           norm_type='BN',
                           depthwise=False,
                           in_channels=in_channels,
                           mask_patch_size=mask_patch_size,
                           mask_ratio=mask_ratio,
                           is_train=is_train,
                           norm_pix_loss=norm_pix_loss,
                           de_num_layers=8,
                           dropout=0.1
                           )

def mae_rtcnet_m(in_channels, mask_patch_size, mask_ratio, is_train=False, norm_pix_loss=False) -> MAE_RTCBackbone:
    return MAE_RTCBackbone(width=0.75,
                           depth=0.67,
                           ratio=1.5,
                           act_type='silu',
                           norm_type='BN',
                           depthwise=False,
                           in_channels=in_channels,
                           mask_patch_size=mask_patch_size,
                           mask_ratio=mask_ratio,
                           is_train=is_train,
                           norm_pix_loss=norm_pix_loss,
                           de_num_layers=8,
                           dropout=0.1
                           )

def mae_rtcnet_l(in_channels, mask_patch_size, mask_ratio, is_train=False, norm_pix_loss=False) -> MAE_RTCBackbone:
    return MAE_RTCBackbone(width=1.0,
                           depth=1.0,
                           ratio=1.0,
                           act_type='silu',
                           norm_type='BN',
                           depthwise=False,
                           in_channels=in_channels,
                           mask_patch_size=mask_patch_size,
                           mask_ratio=mask_ratio,
                           is_train=is_train,
                           norm_pix_loss=norm_pix_loss,
                           de_num_layers=8,
                           dropout=0.1
                           )

def mae_rtcnet_x(in_channels, mask_patch_size, mask_ratio, is_train=False, norm_pix_loss=False) -> MAE_RTCBackbone:
    return MAE_RTCBackbone(width=1.25,
                           depth=1.34,
                           ratio=1.0,
                           act_type='silu',
                           norm_type='BN',
                           depthwise=False,
                           in_channels=in_channels,
                           mask_patch_size=mask_patch_size,
                           mask_ratio=mask_ratio,
                           is_train=is_train,
                           norm_pix_loss=norm_pix_loss,
                           de_num_layers=8,
                           dropout=0.1
                           )


if __name__ == '__main__':
    import torch
    import cv2
    import numpy as np
    from thop import profile

    # build model
    bs, c, h, w = 2, 3, 224, 224
    is_train = False
    x = torch.randn(bs, c, h, w)
    model = mae_rtcnet_l(in_channels=3, mask_patch_size=16, mask_ratio=0.75, is_train=is_train)

    # inference
    outputs = model(x)

    # compute FLOPs & Params
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

    x_preds = outputs["x_pred"]
    masks = outputs["mask"]
    if "loss" in outputs:
        print("Loss: ", outputs["loss"].item())

    with torch.no_grad():
        for bi in range(bs):
            img = x[bi].permute(1, 2, 0).numpy().astype(np.uint8) * 255
            x_pred = x_preds[bi].permute(1, 2, 0).numpy().astype(np.uint8)
            mask = masks[bi].permute(1, 2, 0).numpy().astype(np.uint8) * 255
            cv2.imshow('masked image', img)
            cv2.waitKey(0)
            cv2.imshow('pred image', x_pred)
            cv2.waitKey(0)
            cv2.imshow('mask', mask)
            cv2.waitKey(0)
