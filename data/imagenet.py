import os
import torch.utils.data as data
import torchvision.transforms as tf
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
    
