import torchvision.transforms as tf


# build transform for CIFAR
def build_cifar_transform(args, is_train=False):
    if is_train:
        transforms = tf.Compose([tf.ToTensor(), tf.Normalize(0.5, 0.5)])
    else:
        transforms = tf.Compose([tf.ToTensor(), tf.Normalize(0.5, 0.5)])

    return transforms

# build transform for ImageNet
def build_imagenet_transform(args, is_train=False):
    if is_train:
        transforms = tf.Compose([
                tf.RandomResizedCrop(args.img_size, scale=[0.2, 1.0], interpolation=3),
                tf.RandomHorizontalFlip(),
                tf.ToTensor(),
                tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    else:
        transforms = tf.Compose([
                tf.Resize(int(256 / 224 * args.img_size)),
                tf.CenterCrop(args.img_size),
                tf.ToTensor(),
                tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    return transforms
