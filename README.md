# MAE
**PyTorch implementation of Masked AutoEncoder**

Due to limited resources, I only test my randomly designed `ViT-Tiny` on the `CIFAR10` dataset. It is not my goal to reproduce MAE perfectly, but my implementation is aligned with the official MAE as much as possible so that users can learn `MAE` quickly and accurately.

## 1. Pretrain
We have kindly provided the bash script `train_pretrain.sh` file for pretraining. You can modify some hyperparameters in the script file according to your own needs.

```Shell
bash train_pretrain.sh
```

## 2. Finetune
We have kindly provided the bash script `train_finetune.sh` file for finetuning. You can modify some hyperparameters in the script file according to your own needs.

```Shell
bash train_finetune.sh
```

## 3. Evaluate 
- Evaluate the `top1 & top5` accuracy of `ViT-Tiny` on CIFAR10 dataset:
```Shell
python train_finetune.py --dataset cifar10 -m vit_tiny --batch_size 256 --img_size 32 --patch_size 2 --eval --resume path/to/vit_tiny_cifar10.pth
```

- Evaluate the `top1 & top5` accuracy of `ViT-Tiny` on ImageNet-1K dataset:
```Shell
python train_finetune.py --dataset imagenet_1k -m vit_tiny --batch_size 256 --img_size 224 --patch_size 16 --eval --resume path/to/vit_tiny_imagenet_1k.pth
```


## 4. Visualize Image Reconstruction
- Evaluate `MAE-ViT-Tiny` on CIFAR10 dataset:
```Shell
python train_pretrain.py --dataset cifar10 -m mae_vit_tiny --resume path/to/mae_vit_tiny_cifar10.pth --img_size 32 --patch_size 2 --eval --batch_size 1
```

- Evaluate `MAE-ViT-Tiny` on ImageNet-1K dataset:
```Shell
python train_pretrain.py --dataset imagenet_1k -m mae_vit_tiny --resume path/to/mae_vit_tiny_imagenet_1k.pth --img_size 224 --patch_size 16 --eval --batch_size 1
```


## 5. Experiments
### 5.1 MAE pretrain
- Visualization on CIFAR10 validation

Masked Image | Original Image | Reconstructed Image

![image](./img_files/visualize_cifar10_mae_vit_nano.png)

- Visualization on ImageNet validation

...


### 5.2 Finutune
- On CIFAR10

|  Model   |  MAE pretrained  | Epoch | Top 1     | Weight |  MAE weight  |
|  :---:   |       :---:      | :---: | :---:     | :---:  |    :---:     |
| ViT-Tiny |        No        | 300   | 86.8      | [ckpt](https://github.com/yjh0410/MAE/releases/download/checkpoints/vit-tiny-scratch-299-Acc1-86.79.pth) | - |
| ViT-Tiny |        Yes       | 100   | **91.8**  | [ckpt](https://github.com/yjh0410/MAE/releases/download/checkpoints/vit-tiny-finetune-99-Acc1-91.78.pth) | [ckpt](https://github.com/yjh0410/MAE/releases/download/checkpoints/vit-tiny-pretrained-799.pth)


- On ImageNet-1K

|  Model   |  MAE pretrained  | Epoch | Top 1 | Weight |  MAE weight  |
|  :---:   |       :---:      | :---: | :---: | :---:  |    :---:     |
| ViT-Tiny |        No        | 300   |       |        | |
| ViT-Tiny |        Yes       | 100   |       |        | |

Since ImageNet-1K is a sufficiently large-scale dataset, we recommend using the default training hyperparameters of the code to pretrain MAE and finetune ViT from MAE pretraining weight. 


## 6. Acknowledgment
Thank you to **Kaiming He** for his inspiring work on [MAE](http://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf). His research effectively elucidates the semantic distinctions between vision and language, offering valuable insights for subsequent vision-related studies. I would also like to express my gratitude for the official source code of [MAE](https://github.com/facebookresearch/mae). Additionally, I appreciate the efforts of [**IcarusWizard**](https://github.com/IcarusWizard) for reproducing the [MAE](https://github.com/IcarusWizard/MAE) implementation.
