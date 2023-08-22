import torch.utils.data as data
import torchvision.transforms as tf
from torchvision.datasets import CIFAR10


class CifarDataset(data.Dataset):
    def __init__(self, is_train=False, transform=None):
        super().__init__()
        # ----------------- basic parameters -----------------
        self.pixel_mean = [0.5, 0.5, 0.5]
        self.pixel_std =  [0.5, 0.5, 0.5]
        self.image_set = 'train' if is_train else 'val'
        # ----------------- dataset & transforms -----------------
        self.transform = transform
        if is_train:
            self.dataset = CIFAR10('data/cifar', train=True, download=True, transform=self.transform)
        else:
            self.dataset = CIFAR10('data/cifar', train=False, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, target = self.dataset[index]

        return image, target
    
