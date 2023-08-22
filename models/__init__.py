from .vit.build import build_vit


def build_model(args):
    if args.model in ['vit_nano', 'vit_tiny', 'vit_base', 'vit_large', 'vit_huge']:
        model = build_vit(args)

    return model
