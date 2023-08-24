# --------------- Finetune on CIFAR ---------------
python mae_finetune.py \
        --cuda \
        --dataset cifar10 \
        -m vit_tiny \
        --batch_size 256 \
        --img_size 32 \
        --patch_size 2 \
        --max_epoch 50 \
        --wp_epoch 5 \
        --mae_pretrained weights/cifar10/mae_vit_tiny/mae_vit_tiny_cifar10.pth

# --------------- Finetune on ImageNet ---------------
# python mae_finetune.py \
#         --cuda \
#         --root path/to/imagenet_1k \
#         --dataset imagenet_1k \
#         -m vit_nano \
#         --batch_size 256 \
#         --img_size 224 \
#         --patch_size 16 \
#         --max_epoch 50 \
#         --wp_epoch 5 \
#         --mae_pretrained weights/imagenet_1k/mae_vit_nano/mae_vit_nano_imagenet_1k.pth