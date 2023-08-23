import torch
from .pos_embed import interpolate_pos_embed


# ------------------------ Vision Transformer ------------------------
from .vit import vit_nano, vit_tiny, vit_base, vit_large, vit_huge

def build_vit(args):
    # build vit model
    if args.model == 'vit_nano':
        model = vit_nano(args.img_size, args.patch_size, args.img_dim, args.num_classes)
    elif args.model == 'vit_tiny':
        model = vit_tiny(args.img_size, args.patch_size, args.img_dim, args.num_classes)
    elif args.model == 'vit_base':
        model = vit_base(args.img_size, args.patch_size, args.img_dim, args.num_classes)
    elif args.model == 'vit_large':
        model = vit_large(args.img_size, args.patch_size, args.img_dim, args.num_classes)
    elif args.model == 'vit_huge':
        model = vit_huge(args.img_size, args.patch_size, args.img_dim, args.num_classes)
    
    # load pretrained
    if args.mae_pretrained is not None:
        ## TODO:
        print('Loading MAE pretrained from <{}> for <{}> ...'.format('mae_'+args.model, args.model))
        checkpoint = torch.load(args.mae_pretrained, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # collect MAE-ViT's encoder weight
        encoder_state_dict = {}
        for k in list(checkpoint_state_dict.keys()):
            if 'mae_encoder' in k and k[12:] in model_state_dict.keys():
                encoder_state_dict[k[12:]] = checkpoint_state_dict[k]

        # interpolate position embedding
        interpolate_pos_embed(model, encoder_state_dict)

        # load encoder weight into ViT's encoder
        model.load_state_dict(encoder_state_dict, strict=False)

    return model


# ------------------------ MAE Vision Transformer ------------------------
from .vit_mae import mae_vit_nano, mae_vit_tiny, mae_vit_base, mae_vit_large, mae_vit_huge

def build_mae_vit(args):
    # build vit model
    if args.model == 'mae_vit_nano':
        model = mae_vit_nano(args.img_size, args.patch_size, args.img_dim, args.mask_ratio, args.norm_pix_loss)
    elif args.model == 'mae_vit_tiny':
        model = mae_vit_tiny(args.img_size, args.patch_size, args.img_dim, args.mask_ratio, args.norm_pix_loss)
    elif args.model == 'mae_vit_base':
        model = mae_vit_base(args.img_size, args.patch_size, args.img_dim, args.mask_ratio, args.norm_pix_loss)
    elif args.model == 'mae_vit_large':
        model = mae_vit_large(args.img_size, args.patch_size, args.img_dim, args.mask_ratio, args.norm_pix_loss)
    elif args.model == 'mae_vit_huge':
        model = mae_vit_huge(args.img_size, args.patch_size, args.img_dim, args.mask_ratio, args.norm_pix_loss)

    return model