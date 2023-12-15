import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10


class CifarDataset(data.Dataset):
    def __init__(self, is_train=False, transform=None):
        super().__init__()
        # ----------------- basic parameters -----------------
        self.pixel_mean = [0.5, 0.5, 0.5]
        self.pixel_std =  [0.5, 0.5, 0.5]
        self.is_train  = is_train
        self.image_set = 'train' if is_train else 'val'
        # ----------------- dataset & transforms -----------------
        self.transform = transform if transform is not None else self.build_transform()
        if is_train:
            self.dataset = CIFAR10('cifar_data/', train=True, download=True, transform=self.transform)
        else:
            self.dataset = CIFAR10('cifar_data/', train=False, download=True, transform=self.transform)

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

    def build_transform(self):
        if self.is_train:
            transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
        else:
            transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

        return transforms

if __name__ == "__main__":
    import cv2
    import torch
    import argparse
    
    parser = argparse.ArgumentParser(description='Cifar-Dataset')

    # opt
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/VOCdevkit/',
                        help='data root')
    parser.add_argument('--img_size', default=224, type=int,
                        help='input image size.')
    args = parser.parse_args()

    def patchify(imgs, patch_size):
        """
        imgs: (B, 3, H, W)
        x:    (B, N, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(x, patch_size):
        """
        x:    (B, N, patch_size**2 *3)
        imgs: (B, 3, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs

    def random_masking(x, mask_patch_size=2, mask_ratio=0.75):
        # ----------------- Step-1: Patch embed -----------------
        # Patchify: [B, C, H, W] -> [B, N, C*P*P]
        H, W = x.shape[2:]
        Hp, Wp = H // mask_patch_size, W // mask_patch_size
        patches = patchify(x, mask_patch_size)
        B, N, C = patches.shape

        # ----------------- Step-2: Random masking -----------------
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)        # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore the original position of each patch

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # [B, N_nomask, 3*P*P]
        keep_patches = torch.gather(patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        # unshuffle to get the masked image [B, N, 3*P*P]
        mask_patches = torch.zeros(B, N-len_keep, C)
        x_masked = torch.cat([keep_patches, mask_patches], dim=1)
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # ----------------- Step-3: Reshape masked patches to image format -----------------
        x_masked = unpatchify(x_masked, mask_patch_size)
        mask = mask.view(B, Hp, Wp)

        return x_masked, mask
  
    # dataset
    dataset = CifarDataset(is_train=True)  
    print('Dataset size: ', len(dataset))

    for i in range(1000):
        image, target = dataset.pull_image(i)
        # to BGR
        image = image[..., (2, 1, 0)]
        print(image.shape)

        cv2.imshow('image', image)
        cv2.waitKey(0)

        image = torch.as_tensor(image).permute(2, 0, 1).unsqueeze(0)
        image = torch.cat([image] * 8, dim=0)
        print(image.shape)

        image_masked, masks = random_masking(image)

        for bi in range(8):
            img = image_masked[bi].permute(1, 2, 0).numpy().astype(np.uint8)
            mask = masks[bi].numpy().astype(np.uint8) * 255
            mask = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('masked image', img)
            cv2.waitKey(0)
            cv2.imshow('mask', mask)
            cv2.waitKey(0)

