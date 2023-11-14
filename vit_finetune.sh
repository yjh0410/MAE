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
        --base_lr 1e-3 \
        --min_lr 1e-6 \
        --layer_decay 0.75 \
        --weight_decay 0.05 \
        --mae_pretrained weights/cifar10/mae_vit_tiny/mae_vit_tiny_epoch_399_0.03.pth

# --------------- Finetune on ImageNet ---------------
# python mae_finetune.py \
#         --cuda \
#         --root path/to/imagenet_1k \
#         --dataset imagenet_1k \
#         -m vit_tiny \
#         --batch_size 256 \
#         --img_size 224 \
#         --patch_size 16 \
#         --max_epoch 100 \
#         --wp_epoch 5 \
#         --base_lr 0.0005 \
#         --weight_decay 0.05 \
#         --mae_pretrained weights/imagenet_1k/mae_vit_tiny/mae_vit_tiny_imagenet_1k.pth