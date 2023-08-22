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
        pass

    return model
