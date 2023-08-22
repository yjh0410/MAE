import torch
import torch.nn as nn

try:
    from vit_modules import ViTBlock
    from pos_embed import *
except:
    from .vit_modules import ViTBlock
    from .pos_embed import *

# ------------------------ Basic Modules ------------------------
## Masked ViT Encoder
class MAE_ViT_Encoder(nn.Module):
    def __init__(self,
                 img_size      :int   = 224,
                 patch_size    :int   = 16,
                 img_dim       :int   = 3,
                 en_emb_dim    :int   = 768,
                 en_num_layers :int   = 12,
                 en_num_heads  :int   = 12,
                 qkv_bias      :bool  = True,
                 mlp_ratio     :float = 4.0,
                 dropout       :float = 0.1,
                 mask_ratio    :float = 0.75):
        super().__init__()
        # -------- basic parameters --------
        self.img_size = img_size
        self.img_dim = img_dim
        self.en_emb_dim = en_emb_dim
        self.en_num_layers = en_num_layers
        self.en_num_heads = en_num_heads
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        # -------- network parameters --------
        ## vit encoder
        self.patch_embed = nn.Conv2d(img_dim, en_emb_dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.ModuleList([ViTBlock(en_emb_dim, qkv_bias, en_num_heads, mlp_ratio, dropout) for _ in range(en_num_layers)])
        self.norm        = nn.LayerNorm(en_emb_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches + 1, en_emb_dim), requires_grad=False)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, en_emb_dim))

        self._init_weights()

    def _init_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

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

    def random_masking(self, x):
        B, N, C = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)        # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore the original position of each patch

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get th binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward(self, x):
        # patch embed
        x = self.patch_embed(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)

        return x, mask, ids_restore

## Masked ViT Decoder
class MAE_ViT_Decoder(nn.Module):
    def __init__(self,
                 img_size      :int   = 16,
                 patch_size    :int   = 16,
                 img_dim       :int   = 3,
                 en_emb_dim    :int   = 784,
                 de_emb_dim    :int   = 512,
                 de_num_layers :int   = 12,
                 de_num_heads  :int   = 12,
                 qkv_bias      :bool  = True,
                 mlp_ratio     :float = 4.0,
                 dropout       :float = 0.1,
                 norm_pix_loss :bool = False):
        super().__init__()
        # -------- basic parameters --------
        self.img_dim = img_dim
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.en_emb_dim = en_emb_dim
        self.de_emb_dim = de_emb_dim
        self.de_num_layers = de_num_layers
        self.de_num_heads = de_num_heads
        self.norm_pix_loss = norm_pix_loss
        # -------- network parameters --------
        self.mask_token        = nn.Parameter(torch.zeros(1, 1, de_emb_dim))
        self.decoder_embed     = nn.Linear(en_emb_dim, de_emb_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, de_emb_dim), requires_grad=False)  # fixed sin-cos embedding
        self.transformer       = nn.ModuleList([ViTBlock(de_emb_dim, qkv_bias, de_num_heads, mlp_ratio, dropout) for _ in range(de_num_layers)])
        self.decoder_norm      = nn.LayerNorm(de_emb_dim)
        self.decoder_pred      = nn.Linear(de_emb_dim, patch_size**2 * img_dim, bias=True)
        
        self._init_weights()

    def _init_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

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

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        B, N_nomask = x.shape[:2]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - (N_nomask - 1), 1)       # [B, N_mask, C], N_mask = (N-1) - N_nomask
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)                                       # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed w/ cls token
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for block in self.transformer:
            x = block(x)
        x = self.decoder_norm(x)

        # predict pixels
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


# ------------------------ MAE Vision Transformer ------------------------
## Masked ViT
class MAE_VisionTransformer(nn.Module):
    def __init__(self,
                 img_size      :int   = 16,
                 patch_size    :int   = 16,
                 img_dim       :int   = 3,
                 en_emb_dim    :int   = 784,
                 de_emb_dim    :int   = 512,
                 en_num_layers :int   = 12,
                 de_num_layers :int   = 12,
                 en_num_heads  :int   = 12,
                 de_num_heads  :int   = 16,
                 qkv_bias      :bool  = True,
                 mlp_ratio     :float = 4.0,
                 dropout       :float = 0.1,
                 mask_ratio    :float = 0.75,
                 norm_pix_loss :bool = False):
        super().__init__()
        # -------- basic parameters --------
        self.img_dim = img_dim
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        ## encoder
        self.en_emb_dim = en_emb_dim
        self.en_num_layers = en_num_layers
        self.en_num_heads = en_num_heads
        self.mask_ratio = mask_ratio
        ## decoder
        self.de_emb_dim = de_emb_dim
        self.de_num_layers = de_num_layers
        self.de_num_heads = de_num_heads
        self.norm_pix_loss = norm_pix_loss
        # -------- network parameters --------
        self.mae_encoder = MAE_ViT_Encoder(
            img_size, patch_size, img_dim, en_emb_dim, en_num_layers, en_num_heads, qkv_bias, mlp_ratio, dropout, mask_ratio)
        self.mae_decoder = MAE_ViT_Decoder(
            img_size, patch_size, img_dim, en_emb_dim, de_emb_dim, de_num_layers, de_num_heads, qkv_bias, mlp_ratio, dropout, norm_pix_loss)


    def forward(self, x):
        x, mask, ids_restore = self.mae_encoder(x)
        x = self.mae_decoder(x, ids_restore)
        output = {
            'x_pred': x,
            'mask': mask
        }

        return output


# ------------------------ Model Functions ------------------------
def mae_vit_nano(img_size=224, patch_size=16, img_dim=3, mask_ratio=0.75, norm_pix_loss=False):
    model = MAE_VisionTransformer(img_size      = img_size,
                                  patch_size    = patch_size,
                                  img_dim       = img_dim,
                                  en_emb_dim    = 192,
                                  de_emb_dim    = 512,
                                  en_num_layers = 12,
                                  de_num_layers = 8,
                                  en_num_heads  = 12,
                                  de_num_heads  = 16,
                                  qkv_bias      = True,
                                  mlp_ratio     = 4.0,
                                  dropout       = 0.1,
                                  mask_ratio    = mask_ratio,
                                  norm_pix_loss = norm_pix_loss)

    return model

def mae_vit_tiny(img_size=224, patch_size=16, img_dim=3, mask_ratio=0.75, norm_pix_loss=False):
    model = MAE_VisionTransformer(img_size      = img_size,
                                  patch_size    = patch_size,
                                  img_dim       = img_dim,
                                  en_emb_dim    = 384,
                                  de_emb_dim    = 512,
                                  en_num_layers = 12,
                                  de_num_layers = 8,
                                  en_num_heads  = 12,
                                  de_num_heads  = 16,
                                  qkv_bias      = True,
                                  mlp_ratio     = 4.0,
                                  dropout       = 0.1,
                                  mask_ratio    = mask_ratio,
                                  norm_pix_loss = norm_pix_loss)

    return model


if __name__ == '__main__':
    import torch
    from ptflops import get_model_complexity_info

    # build model
    model = mae_vit_tiny(patch_size=16)

    # calculate params & flops
    flops_count, params_count = get_model_complexity_info(model,(3,224,224), as_strings=True, print_per_layer_stat=False)

    print('flops: ', flops_count)
    print('params: ', params_count)
    