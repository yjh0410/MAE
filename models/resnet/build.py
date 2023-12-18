import os
import torch


# ------------------------ ResNet ------------------------
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

def build_resnet(args):
    # build vit model
    if args.model == 'resnet18':
        model = resnet18()
    elif args.model == 'resnet34':
        model = resnet34()
    elif args.model == 'resnet50':
        model = resnet50()
    elif args.model == 'resnet101':
        model = resnet101()
    elif args.model == 'resnet152':
        model = resnet152()
    
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
from .resnet_mae import mae_resnet18, mae_resnet34, mae_resnet50, mae_resnet101, mae_resnet152

def build_mae_resnet(args, is_train=False):
    # build vit model
    if args.model == 'mae_resnet18':
        model = mae_resnet18(mask_patch_size=args.patch_size, mask_ratio=args.mask_ratio, is_train=is_train, norm_pix_loss=args.norm_pix_loss)
    elif args.model == 'mae_resnet34':
        model = mae_resnet34(mask_patch_size=args.patch_size, mask_ratio=args.mask_ratio, is_train=is_train, norm_pix_loss=args.norm_pix_loss)
    elif args.model == 'mae_resnet50':
        model = mae_resnet50(mask_patch_size=args.patch_size, mask_ratio=args.mask_ratio, is_train=is_train, norm_pix_loss=args.norm_pix_loss)
    elif args.model == 'mae_resnet101':
        model = mae_resnet101(mask_patch_size=args.patch_size, mask_ratio=args.mask_ratio, is_train=is_train, norm_pix_loss=args.norm_pix_loss)
    elif args.model == 'mae_resnet152':
        model = mae_resnet152(mask_patch_size=args.patch_size, mask_ratio=args.mask_ratio, is_train=is_train, norm_pix_loss=args.norm_pix_loss)

    return model