import torch

from .vit.build import build_vit, build_mae_vit
from .aim.build import build_aim, build_aim_cls
from .resnet.build import build_resnet, build_mae_resnet
from .rtcnet.build import build_rtcnet, build_mae_rtcnet


def build_model(args, is_train=False):
    # --------------------------- ViT series ---------------------------
    if   args.model in ['vit_nano', 'vit_tiny', 'vit_base', 'vit_large', 'vit_huge']:
        model = build_vit(args)

    elif args.model in ['mae_vit_nano', 'mae_vit_tiny', 'mae_vit_base', 'mae_vit_large', 'mae_vit_huge']:
        model = build_mae_vit(args, is_train)

    # --------------------------- AIM series ---------------------------
    elif args.model in ['aim_nano', 'aim_tiny', 'aim_base', 'aim_large', 'aim_huge']:
        model = build_aim(args, is_train)

    elif args.model in ['aim_cls_nano', 'aim_cls_tiny', 'aim_cls_base', 'aim_cls_large', 'aim_cls_huge']:
        model = build_aim_cls(args)

    # --------------------------- ResNet series ---------------------------
    elif args.model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        model = build_resnet(args)

    elif args.model in ['mae_resnet18', 'mae_resnet34', 'mae_resnet50', 'mae_resnet101', 'mae_resnet152']:
        model = build_mae_resnet(args, is_train)

    # --------------------------- RTCNet series ---------------------------
    elif args.model in ['rtcnet_n', 'rtcnet_t', 'rtcnet_s', 'rtcnet_m', 'rtcnet_l', 'rtcnet_x']:
        model = build_rtcnet(args)

    elif args.model in ['mae_rtcnet_n', 'mae_rtcnet_t', 'mae_rtcnet_s', 'mae_rtcnet_m', 'mae_rtcnet_l', 'mae_rtcnet_x']:
        model = build_mae_rtcnet(args, is_train)


    if args.resume is not None:
        print('loading trained weight for <{}> from <{}>: '.format(args.model, args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)

    return model
