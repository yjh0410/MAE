# MAE
PyTorch implementation of Masked AutoEncoder


## 1. MAE pretrain
- Train `MAE-ViT-Nano` on CIFAR10 dataset:

```Shell
python mae_pretrain.py --dataset cifar10 -m mae_vit_nano --batch_size 256 --img_size 32 --patch_size 2
```

- Train `MAE-ViT-Nano` on ImageNet dataset:

```Shell
python mae_finetune.py --dataset imagenet -m mae_vit_nano --batch_size 256 --img_size 224 --patch_size 16
```

## 2. Train from scratch
- Train `ViT-Nano` on CIFAR10 dataset:

```Shell
python mae_finetune.py --dataset cifar10 -m vit_nano --batch_size 256 --img_size 32 --patch_size 2
```

- Train `ViT-Nano` on ImageNet dataset:

```Shell
python mae_finetune.py --dataset imagenet -m vit_nano --batch_size 256 --img_size 224 --patch_size 16
```

## 3. Train from MAE pretrained
- Train `ViT-Nano` on CIFAR10 dataset:

```Shell
python mae_finetune.py --dataset cifar10 -m vit_nano --batch_size 256 --img_size 32 --patch_size 2 --mae_pretrained
```

- Train `ViT-Nano` on ImageNet dataset:

```Shell
python mae_finetune.py --dataset imagenet -m vit_nano --batch_size 256 --img_size 224 --patch_size 16 --mae_pretrained
```

## 4. Experiments
### 4.1 MAE pretrain
- Visualization on CIFAR10

...

- Visualization on ImageNet

...


### 4.2 Finutune
- On CIFAR10

|  Model   |  MAE pretrained  | Epoch | Top 1 | Top 5 | Weight |
|  :---:   |       :---:      | :---: | :---: | :---: | :---:  |
| ViT-Nano |        No        | 200   |       |       |        |
| ViT-Nano |        Yes       | 50    |       |       |        |

- On ImageNet-1K

|  Model   |  MAE pretrained  | Epoch | Top 1 | Top 5 | Weight |
|  :---:   |       :---:      | :---: | :---: | :---: | :---:  |
| ViT-Nano |        No        | 200   |       |       |        |
| ViT-Nano |        Yes       | 50    |       |       |        |
