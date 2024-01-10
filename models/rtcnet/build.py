import os
import torch


# ------------------------ ResNet ------------------------
from .rtcnet import rtcnet_n, rtcnet_t, rtcnet_s, rtcnet_m, rtcnet_l, rtcnet_x

def build_rtcnet(args):
    # build vit model
    if args.model == 'rtcnet_n':
        model = rtcnet_n(args.num_classes)
    elif args.model == 'rtcnet_t':
        model = rtcnet_t(args.num_classes)
    elif args.model == 'rtcnet_s':
        model = rtcnet_s(args.num_classes)
    elif args.model == 'rtcnet_m':
        model = rtcnet_m(args.num_classes)
    elif args.model == 'rtcnet_l':
        model = rtcnet_l(args.num_classes)
    elif args.model == 'rtcnet_x':
        model = rtcnet_x(args.num_classes)
    
    # load pretrained
    if args.mae_pretrained is not None:
        # check path
        if not os.path.exists(args.mae_pretrained):
            print("No mae pretrained model.")
            return model
        ## load mae pretrained model
        print('Loading MAE pretrained from <{}> for <{}> ...'.format('mae_'+args.model, args.model))
        checkpoint = torch.load(args.mae_pretrained, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # collect MAE-ViT's encoder weight
        reformat_state_dict = {}
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict.keys():
                reformat_state_dict[k] = checkpoint_state_dict[k]

        # load encoder weight into ViT's encoder
        model.load_state_dict(reformat_state_dict, strict=False)

    return model


# ------------------------ MAE ResNet ------------------------
from .rtcnet_mae import mae_rtcnet_n, mae_rtcnet_t, mae_rtcnet_s, mae_rtcnet_m, mae_rtcnet_l, mae_rtcnet_x

def build_mae_rtcnet(args, is_train=False):
    # build vit model
    if args.model == 'mae_rtcnet_n':
        model = mae_rtcnet_n(in_channels=args.img_dim, mask_patch_size=args.patch_size, mask_ratio=args.mask_ratio, is_train=is_train, norm_pix_loss=args.norm_pix_loss)
    elif args.model == 'mae_rtcnet_t':
        model = mae_rtcnet_t(in_channels=args.img_dim, mask_patch_size=args.patch_size, mask_ratio=args.mask_ratio, is_train=is_train, norm_pix_loss=args.norm_pix_loss)
    elif args.model == 'mae_rtcnet_s':
        model = mae_rtcnet_s(in_channels=args.img_dim, mask_patch_size=args.patch_size, mask_ratio=args.mask_ratio, is_train=is_train, norm_pix_loss=args.norm_pix_loss)
    elif args.model == 'mae_rtcnet_m':
        model = mae_rtcnet_m(in_channels=args.img_dim, mask_patch_size=args.patch_size, mask_ratio=args.mask_ratio, is_train=is_train, norm_pix_loss=args.norm_pix_loss)
    elif args.model == 'mae_rtcnet_l':
        model = mae_rtcnet_l(in_channels=args.img_dim, mask_patch_size=args.patch_size, mask_ratio=args.mask_ratio, is_train=is_train, norm_pix_loss=args.norm_pix_loss)
    elif args.model == 'mae_rtcnet_x':
        model = mae_rtcnet_x(in_channels=args.img_dim, mask_patch_size=args.patch_size, mask_ratio=args.mask_ratio, is_train=is_train, norm_pix_loss=args.norm_pix_loss)

    return model
