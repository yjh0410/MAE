import torch

from .vit.build import build_vit, build_mae_vit
from .resnet.build import build_resnet, build_mae_resnet


def build_model(args):
    # --------------------------- ViT series ---------------------------
    if args.model in ['vit_nano', 'vit_tiny', 'vit_base', 'vit_large', 'vit_huge']:
        model = build_vit(args)

    elif args.model in ['mae_vit_nano', 'mae_vit_tiny', 'mae_vit_base', 'mae_vit_large', 'mae_vit_huge']:
        model = build_mae_vit(args)

    # --------------------------- ResNet series ---------------------------
    elif args.model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        model = build_resnet(args)

    elif args.model in ['mae_resnet18', 'mae_resnet34', 'mae_resnet50', 'mae_resnet101', 'mae_resnet152']:
        model = build_mae_resnet(args)


    if args.resume is not None:
        print('loading trained weight for <{}> from <{}>: '.format(args.model, args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)

    return model
