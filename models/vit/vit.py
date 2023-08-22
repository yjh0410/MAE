import torch
import torch.nn as nn

try:
    from vit_modules import ViTBlock
except:
    from .vit_modules import ViTBlock


# ------------------------ Vision Transformer ------------------------
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size    :int   = 224,
                 patch_size  :int   = 16,
                 img_dim     :int   = 3,
                 emb_dim     :int   = 768,
                 num_layers  :int   = 12,
                 num_heads   :int   = 12,
                 qkv_bias    :bool  = True,
                 mlp_ratio   :float = 4.0,
                 dropout     :float = 0.1,
                 num_classes :int   = 1000):
        super().__init__()
        # -------- basic parameters --------
        self.img_size = img_size
        self.img_dim = img_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_patches = (img_size // patch_size) ** 2
        # -------- network parameters --------
        ## vit encoder
        self.patch_embed = nn.Conv2d(img_dim, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.ModuleList([ViTBlock(emb_dim, qkv_bias, num_heads, mlp_ratio, dropout) for _ in range(num_layers)])
        self.norm        = nn.LayerNorm(emb_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, emb_dim))
        ## classifier
        self.classifier  = nn.Linear(emb_dim, num_classes)

    def _init_weight(self,):
        # initialize cls_token
        nn.init.normal_(self.cls_token, std=1e-6)
        # initialize pos_embed
        nn.init.normal_(self.pos_embed, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, x):
        # patch embed
        x = self.patch_embed(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        # transformer
        x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)
        x += self.pos_embed
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)

        # classify
        return self.classifier(x[:, 0, :])


# ------------------------ Model Functions ------------------------
def vit_nano(img_size=224, patch_size=16, img_dim=3, num_classes=1000):
    model = VisionTransformer(img_size    = img_size,
                              patch_size  = patch_size,
                              img_dim     = img_dim,
                              emb_dim     = 192,
                              num_layers  = 12,
                              num_heads   = 8,
                              mlp_ratio   = 4.0,
                              dropout     = 0.1,
                              num_classes = num_classes)

    return model

def vit_tiny(img_size=224, patch_size=16, img_dim=3, num_classes=1000):
    model = VisionTransformer(img_size    = img_size,
                              patch_size  = patch_size,
                              img_dim     = img_dim,
                              emb_dim     = 384,
                              num_layers  = 12,
                              num_heads   = 8,
                              mlp_ratio   = 4.0,
                              dropout     = 0.1,
                              num_classes = num_classes)

    return model

def vit_base(img_size=224, patch_size=16, img_dim=3, num_classes=1000):
    model = VisionTransformer(img_size    = img_size,
                              patch_size  = patch_size,
                              img_dim     = img_dim,
                              emb_dim     = 768,
                              num_layers  = 12,
                              num_heads   = 12,
                              mlp_ratio   = 4.0,
                              dropout     = 0.1,
                              num_classes = num_classes)

    return model

def vit_large(img_size=224, patch_size=16, img_dim=3, num_classes=1000):
    model = VisionTransformer(img_size    = img_size,
                              patch_size  = patch_size,
                              img_dim     = img_dim,
                              emb_dim     = 1024,
                              num_layers  = 24,
                              num_heads   = 16,
                              mlp_ratio   = 4.0,
                              dropout     = 0.1,
                              num_classes = num_classes)

    return model

def vit_huge(img_size=224, patch_size=16, img_dim=3, num_classes=1000):
    model = VisionTransformer(img_size    = img_size,
                              patch_size  = patch_size,
                              img_dim     = img_dim,
                              emb_dim     = 1280,
                              num_layers  = 32,
                              num_heads   = 16,
                              mlp_ratio   = 4.0,
                              dropout     = 0.1,
                              num_classes = num_classes)

    return model


if __name__ == '__main__':
    import torch
    from ptflops import get_model_complexity_info

    # build model
    model = vit_tiny(patch_size=16, mae_pretrained=True)

    # calculate params & flops
    flops_count, params_count = get_model_complexity_info(model,(3,224,224), as_strings=True, print_per_layer_stat=False)

    print('flops: ', flops_count)
    print('params: ', params_count)
    