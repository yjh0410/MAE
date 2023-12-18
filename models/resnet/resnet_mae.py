from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .resnet_modules import conv1x1, BasicBlock, Bottleneck
except:
     from resnet_modules import conv1x1, BasicBlock, Bottleneck
   

class MAE_ResNet(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 zero_init_residual: bool = False,
                 in_channels: int = 3,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 mask_patch_size: int = 32,
                 mask_ratio: float = 0.75,
                 is_train: bool = False,
                 norm_pix_loss: bool = False
                 ) -> None:
        super().__init__()
        # --------------- Basic parameters ----------------
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64
        self.dilation = 1
        self.is_train = is_train
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        self.norm_pix_loss = norm_pix_loss
        self.zero_init_residual = zero_init_residual
        self.replace_stride_with_dilation = [False, False, False] if replace_stride_with_dilation is None else replace_stride_with_dilation
        self.out_stride = 16 if self.replace_stride_with_dilation[-1] else 32
        if len(self.replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {self.replace_stride_with_dilation}"
            )
        
        # --------------- Network parameters ----------------
        self._norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        ## Stem layer
        self.conv1   = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = self._norm_layer(self.inplanes)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ## Res Layer
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=self.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=self.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=self.replace_stride_with_dilation[2])
        ## Out layer
        self.output_proj = nn.Conv2d(512 * block.expansion, in_channels * self.out_stride ** 2, kernel_size=1)

        self._init_layer()

    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False,
                    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

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
        H, W = x.shape[2:]
        Hp, Wp = H // self.mask_patch_size, W // self.mask_patch_size
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
        mask_patches = torch.zeros(B, N-len_keep, C)
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

    def forward(self, x: Tensor) -> Tensor:
        imgs = x
        x_masked, mask = self.random_masking(x)

        # See note [TorchScript super()]
        x = self.conv1(x_masked)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.output_proj(x)

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
def _resnet(block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], **kwargs) -> MAE_ResNet:
    return MAE_ResNet(block, layers, **kwargs)


def mae_resnet18(**kwargs) -> MAE_ResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def mae_resnet34(**kwargs) -> MAE_ResNet:
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def mae_resnet50(**kwargs) -> MAE_ResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def mae_resnet101(**kwargs) -> MAE_ResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def mae_resnet152(**kwargs) -> MAE_ResNet:
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    import torch
    import cv2
    import numpy as np
    from thop import profile

    # build model
    bs, c, h, w = 2, 3, 224, 224
    is_train = True
    x = torch.randn(bs, c, h, w)
    model = mae_resnet50(mask_patch_size=16, is_train=is_train)

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
