import os
import torch


# ------------------------ Vision Transformer ------------------------
from .aim import aim_cls_nano, aim_cls_tiny, aim_cls_base, aim_cls_large, aim_cls_huge

def build_aim_cls(args):
    # build vit model
    if   args.model == 'aim_cls_nano':
        model = aim_cls_nano(args.img_size, args.patch_size, args.img_dim, args.num_classes)
    elif args.model == 'aim_cls_tiny':
        model = aim_cls_tiny(args.img_size, args.patch_size, args.img_dim, args.num_classes)
    elif args.model == 'aim_cls_base':
        model = aim_cls_base(args.img_size, args.patch_size, args.img_dim, args.num_classes)
    elif args.model == 'aim_cls_large':
        model = aim_cls_large(args.img_size, args.patch_size, args.img_dim, args.num_classes)
    elif args.model == 'aim_cls_huge':
        model = aim_cls_huge(args.img_size, args.patch_size, args.img_dim, args.num_classes)
    
    # load pretrained
    if args.pretrained is not None:
        # check path
        if not os.path.exists(args.pretrained):
            print("No AIM pretrained model.")
            return model
        ## load mae pretrained model
        print('Loading AIM pretrained from <{}> for <{}> ...'.format('aim_'+args.model[8:], args.model))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # collect AIM's encoder weight
        encoder_state_dict = {}
        for k in list(checkpoint_state_dict.keys()):
            if 'aim_encoder' in k and k[12:] in model_state_dict.keys():
                encoder_state_dict[k[12:]] = checkpoint_state_dict[k]

        # load encoder weight into ViT's encoder
        model.load_state_dict(encoder_state_dict, strict=False)

    return model




# ------------------------ Autoregressive Image Models ------------------------
from .aim import aim_nano, aim_tiny, aim_base, aim_large, aim_huge

def build_aim(args, is_train=False):
    # build vit model
    if   args.model == 'aim_nano':
        model = aim_nano(args.img_size, args.patch_size, args.img_dim, is_train, args.norm_pix_loss)
    elif args.model == 'aim_tiny':
        model = aim_tiny(args.img_size, args.patch_size, args.img_dim, is_train, args.norm_pix_loss)
    elif args.model == 'aim_base':
        model = aim_base(args.img_size, args.patch_size, args.img_dim, is_train, args.norm_pix_loss)
    elif args.model == 'aim_large':
        model = aim_large(args.img_size, args.patch_size, args.img_dim, is_train, args.norm_pix_loss)
    elif args.model == 'aim_huge':
        model = aim_huge(args.img_size, args.patch_size, args.img_dim, is_train, args.norm_pix_loss)

    return model