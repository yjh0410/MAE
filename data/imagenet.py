import os
import numpy as np
import torch.utils.data as data
from torchvision.datasets import ImageFolder


class ImageNet1KDataset(data.Dataset):
    def __init__(self, root, is_train=False, transform=None):
        super().__init__()
        # ----------------- basic parameters -----------------
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        self.image_set = 'train' if is_train else 'val'
        self.data_path = os.path.join(root, self.image_set)
        # ----------------- dataset & transforms -----------------
        self.transform = transform
        self.dataset = ImageFolder(root=self.data_path, transform=self.transform)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, target = self.dataset[index]

        return image, target
    
    def pull_image(self, index):
        # laod data
        image, target = self.dataset[index]

        # denormalize image
        image = image.permute(1, 2, 0).numpy()
        image = (image * self.pixel_std + self.pixel_mean) * 255.
        image = image.astype(np.uint8)
        image = image.copy()

        return image, target


if __name__ == "__main__":
    import cv2
    import argparse
    from transforms import build_imagenet_transform
    
    parser = argparse.ArgumentParser(description='Cifar-Dataset')

    # opt
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset/imagenet/',
                        help='data root')
    parser.add_argument('--img_size', default=224, type=int,
                        help='input image size.')
    args = parser.parse_args()

    # transform
    transform = build_imagenet_transform(args, is_train=True)

    # dataset
    dataset = ImageNet1KDataset(root=args.root, is_train=True, transform=transform)  
    print('Dataset size: ', len(dataset))

    for i in range(1000):
        image, target = dataset.pull_image(i)
        # to BGR
        image = image[..., (2, 1, 0)]
        print(image.shape)

        cv2.imshow('image', image)
        cv2.waitKey(0)