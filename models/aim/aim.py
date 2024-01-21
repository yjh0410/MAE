import math
import torch
import torch.nn as nn

try:
    from aim_modules  import ViTBlock, MLP, AttentionPoolingClassifier
except:
    from .aim_modules import ViTBlock, MLP, AttentionPoolingClassifier


# ------------------------ Model Modules ------------------------
class AIM_Encoder(nn.Module):
    def __init__(self,
                 img_size      :int   = 224,
                 patch_size    :int   = 16,
                 img_dim       :int   = 3,
                 emb_dim       :int   = 768,
                 num_layers    :int   = 12,
                 num_heads     :int   = 12,
                 qkv_bias      :bool  = True,
                 mlp_ratio     :float = 4.0,
                 dropout       :float = 0.1,
                 prefix_causal_mask: bool = False,
                 ):
        super().__init__()
        # -------- basic parameters --------
        self.img_size = img_size
        self.img_dim = img_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2
        self.prefix_causal_mask = prefix_causal_mask
        # -------- network parameters --------
        self.patch_embed = nn.Conv2d(img_dim, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim), requires_grad=False)
        self.norm_layer  = nn.LayerNorm(emb_dim)
        self.transformer = nn.ModuleList([ViTBlock(emb_dim, qkv_bias, num_heads, self.num_patches, mlp_ratio, prefix_causal_mask, dropout)
                                          for _ in range(num_layers)])

        self._init_weights()

    def _init_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = self.get_posembed(self.pos_embed.shape[-1], int(self.num_patches**.5))
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        for m in self.modules():           
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def get_posembed(self, embed_dim, grid_size, temperature=10000):
        scale = 2 * math.pi
        grid_h, grid_w = grid_size, grid_size
        num_pos_feats = embed_dim // 2
        # get grid
        y_embed, x_embed = torch.meshgrid([torch.arange(grid_h, dtype=torch.float32),
                                           torch.arange(grid_w, dtype=torch.float32)])
        # normalize grid coords
        y_embed = y_embed / (grid_h + 1e-6) * scale
        x_embed = x_embed / (grid_w + 1e-6) * scale
    
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[..., None], dim_t)
        pos_y = torch.div(y_embed[..., None], dim_t)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        # [H, W, C] -> [N, C]
        pos_embed = torch.cat((pos_y, pos_x), dim=-1).view(-1, embed_dim)

        return pos_embed.unsqueeze(0)

    def forward(self, x, mask=None):
        # Patch embed
        x = self.patch_embed(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        # Add pos embed
        x = x + self.pos_embed

        # Apply Transformer blocks
        for block in self.transformer:
            x = block(x, mask)
        x = self.norm_layer(x)

        return x

class AIM_Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, mlp_ratio, num_blocks, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = round(in_dim * mlp_ratio)
        self.pixel_decoder = nn.Sequential(*[MLP(in_dim, self.hidden_dim, in_dim, dropout) for _ in range(num_blocks)])
        self.pixel_predictor = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.pixel_decoder(x)
        x = self.pixel_predictor(x)

        return x


# ------------------------ Model ------------------------
## For pretraining
class AIM(nn.Module):
    def __init__(self,
                 img_size      :int   = 16,
                 patch_size    :int   = 16,
                 img_dim       :int   = 3,
                 emb_dim       :int   = 784,
                 num_heads     :int   = 12,
                 num_layers    :int   = 12,
                 num_blocks    :int   = 12,
                 qkv_bias      :bool  = True,
                 mlp_ratio     :float = 4.0,
                 dropout       :float = 0.1,
                 is_train      :bool  = False,
                 norm_pix_loss :bool  = False,
                 prefix_causal_mask :bool = False):
        super().__init__()
        # -------- basic parameters --------
        self.img_dim = img_dim
        self.is_train = is_train
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        ## Encoder
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_causal_mask = prefix_causal_mask
        ## Decoder
        self.num_blocks = num_blocks
        self.norm_pix_loss = norm_pix_loss
        # -------- network parameters --------
        self.aim_encoder = AIM_Encoder(img_size, patch_size, img_dim, emb_dim, num_layers, num_heads, qkv_bias, mlp_ratio, dropout, prefix_causal_mask)
        self.aim_decoder = AIM_Decoder(emb_dim, patch_size**2 * img_dim, mlp_ratio, num_blocks, dropout)

    def patchify(self, imgs, patch_size):
        """
        imgs: (B, 3, H, W)
        x: (N, L, patch_size**2 *3)
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
        x: (B, N, patch_size**2 *3)
        imgs: (B, 3, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs

    def compute_loss(self, x, output):
        """
        imgs: [B, 3, H, W]
        pred: [B, N, C], C = p*p*3
        mask: [B, N], 0 is keep, 1 is remove, 
        """
        target = self.patchify(x, self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        pred, mask = output["x_pred"], output["mask"]
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        return loss

    def forward(self, x, mask):
        """
        Inputs:
            x: (torch.Tensor) -> [B, C, H, W]. Input image.
            mask:  (torch.Tensor) -> [B, N]. prefix mask, where 0 is the retained patch and 1 is the masked patch.
        """
        imgs = x
        x = self.aim_encoder(x, mask)
        x = self.aim_decoder(x)
        output = {
            'x_pred': x,
            'mask': mask
        }

        if self.is_train:
            loss = self.compute_loss(imgs, output)
            output["loss"] = loss

        return output

## For classification
class AIMForImageClassification(nn.Module):
    def __init__(self,
                 img_size      :int   = 16,
                 patch_size    :int   = 16,
                 img_dim       :int   = 3,
                 emb_dim       :int   = 784,
                 num_heads     :int   = 12,
                 num_layers    :int   = 12,
                 num_classes   :int   = 1000,
                 qkv_bias      :bool  = True,
                 mlp_ratio     :float = 4.0,
                 dropout       :float = 0.1,
                 ):
        super().__init__()
        # -------- basic parameters --------
        self.img_dim = img_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.num_classes = num_classes
        ## Encoder
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        # -------- network parameters --------
        self.encoder = AIM_Encoder(img_size, patch_size, img_dim, emb_dim, num_layers, num_heads, qkv_bias, mlp_ratio, dropout, False)
        self.classifier = AttentionPoolingClassifier(emb_dim, num_classes, num_heads, qkv_bias, num_queries=1)

    def forward(self, x):
        """
        Inputs:
            x: (torch.Tensor) -> [B, C, H, W]. Input image.
        """
        x = self.encoder(x)
        x, x_cls = self.classifier(x)

        return x


# ------------------------ Model Functions for Pretraining ------------------------
def aim_nano(img_size=224, patch_size=16, img_dim=3, is_train=False, norm_pix_loss=False):
    model = AIM(img_size      = img_size,
                patch_size    = patch_size,
                img_dim       = img_dim,
                emb_dim       = 192,
                num_layers    = 12,
                num_blocks    = 8,
                num_heads     = 12,
                qkv_bias      = True,
                mlp_ratio     = 4.0,
                dropout       = 0.1,
                is_train      = is_train,
                norm_pix_loss = norm_pix_loss,
                prefix_causal_mask = True)

    return model

def aim_tiny(img_size=224, patch_size=16, img_dim=3, is_train=False, norm_pix_loss=False):
    model = AIM(img_size      = img_size,
                patch_size    = patch_size,
                img_dim       = img_dim,
                emb_dim       = 384,
                num_layers    = 12,
                num_blocks    = 8,
                num_heads     = 12,
                qkv_bias      = True,
                mlp_ratio     = 4.0,
                dropout       = 0.1,
                is_train      = is_train,
                norm_pix_loss = norm_pix_loss,
                prefix_causal_mask = True)

    return model

def aim_base(img_size=224, patch_size=16, img_dim=3, is_train=False, norm_pix_loss=False):
    model = AIM(img_size      = img_size,
                patch_size    = patch_size,
                img_dim       = img_dim,
                emb_dim       = 768,
                num_layers    = 12,
                num_blocks    = 8,
                num_heads     = 12,
                qkv_bias      = True,
                mlp_ratio     = 4.0,
                dropout       = 0.1,
                is_train      = is_train,
                norm_pix_loss = norm_pix_loss,
                prefix_causal_mask = True)

    return model

def aim_large(img_size=224, patch_size=16, img_dim=3, is_train=False, norm_pix_loss=False):
    model = AIM(img_size      = img_size,
                patch_size    = patch_size,
                img_dim       = img_dim,
                emb_dim       = 1024,
                num_layers    = 24,
                num_blocks    = 8,
                num_heads     = 16,
                qkv_bias      = True,
                mlp_ratio     = 4.0,
                dropout       = 0.1,
                is_train      = is_train,
                norm_pix_loss = norm_pix_loss,
                prefix_causal_mask = True)

    return model

def aim_huge(img_size=224, patch_size=16, img_dim=3, is_train=False, norm_pix_loss=False):
    model = AIM(img_size      = img_size,
                patch_size    = patch_size,
                img_dim       = img_dim,
                emb_dim       = 1280,
                num_layers    = 32,
                num_blocks    = 8,
                num_heads     = 16,
                qkv_bias      = True,
                mlp_ratio     = 4.0,
                dropout       = 0.1,
                is_train      = is_train,
                norm_pix_loss = norm_pix_loss,
                prefix_causal_mask = True)

    return model


# ------------------------ Model Functions for Classification ------------------------
def aim_cls_nano(img_size=224, patch_size=16, img_dim=3, num_classes=1000):
    model = AIMForImageClassification(img_size      = img_size,
                                      patch_size    = patch_size,
                                      img_dim       = img_dim,
                                      emb_dim       = 192,
                                      num_heads     = 12,
                                      num_layers    = 12,
                                      num_classes   = num_classes,
                                      qkv_bias      = True,
                                      mlp_ratio     = 4.0,
                                      dropout       = 0.1)

    return model

def aim_cls_tiny(img_size=224, patch_size=16, img_dim=3, num_classes=1000):
    model = AIMForImageClassification(img_size      = img_size,
                                      patch_size    = patch_size,
                                      img_dim       = img_dim,
                                      emb_dim       = 384,
                                      num_heads     = 12,
                                      num_layers    = 12,
                                      num_classes   = num_classes,
                                      qkv_bias      = True,
                                      mlp_ratio     = 4.0,
                                      dropout       = 0.1)

    return model

def aim_cls_base(img_size=224, patch_size=16, img_dim=3, num_classes=1000):
    model = AIMForImageClassification(img_size      = img_size,
                                      patch_size    = patch_size,
                                      img_dim       = img_dim,
                                      emb_dim       = 768,
                                      num_heads     = 12,
                                      num_layers    = 12,
                                      num_classes   = num_classes,
                                      qkv_bias      = True,
                                      mlp_ratio     = 4.0,
                                      dropout       = 0.1)

    return model

def aim_cls_large(img_size=224, patch_size=16, img_dim=3, num_classes=1000):
    model = AIMForImageClassification(img_size      = img_size,
                                      patch_size    = patch_size,
                                      img_dim       = img_dim,
                                      emb_dim       = 1024,
                                      num_heads     = 16,
                                      num_layers    = 24,
                                      num_classes   = num_classes,
                                      qkv_bias      = True,
                                      mlp_ratio     = 4.0,
                                      dropout       = 0.1)

    return model

def aim_cls_huge(img_size=224, patch_size=16, img_dim=3, num_classes=1000):
    model = AIMForImageClassification(img_size      = img_size,
                                      patch_size    = patch_size,
                                      img_dim       = img_dim,
                                      emb_dim       = 1280,
                                      num_heads     = 16,
                                      num_layers    = 32,
                                      num_classes   = num_classes,
                                      qkv_bias      = True,
                                      mlp_ratio     = 4.0,
                                      dropout       = 0.1)

    return model



if __name__ == '__main__':
    import torch
    from thop import profile

    print('===============  AIM pipeline  ===============')
    # parameters
    is_train = True
    img_size = 224
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2

    # generate input data
    x = torch.randn(2, 3, img_size, img_size)
    mask = torch.ones(2, num_patches, dtype=torch.int)
    mask[:, :20] = 0
    model = aim_tiny(img_size, patch_size, 3, is_train, norm_pix_loss=False)

    # inference
    outputs = model(x, mask)
    if "loss" in outputs:
        print("Loss: ", outputs["loss"].item())

    # compute FLOPs & Params
    print('==============================')
    x = torch.randn(1, 3, img_size, img_size)
    mask = torch.zeros(1, num_patches, dtype=torch.int)
    flops, params = profile(model, inputs=(x, mask), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))


    print('===============  AIM-Cls pipeline  ===============')
    # parameters
    img_size = 224
    patch_size = 16
    num_classes = 1000
    num_patches = (img_size // patch_size) ** 2

    # generate input data
    x = torch.randn(2, 3, img_size, img_size)
    model = aim_cls_tiny(img_size, patch_size, 3, num_classes)

    # inference
    outputs = model(x)

    # compute FLOPs & Params
    print('==============================')
    x = torch.randn(1, 3, img_size, img_size)
    mask = torch.zeros(1, num_patches, dtype=torch.int)
    flops, params = profile(model, inputs=(x,), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
