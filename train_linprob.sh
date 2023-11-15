# --------------- Finetune on CIFAR ---------------
python mae_finetune.py \
        --cuda \
        --dataset cifar10 \
        -m vit_tiny \
        --learnable_pos \
        --batch_size 256 \
        --img_size 32 \
        --patch_size 2 \
        --max_epoch 200 \
        --wp_epoch 20 \
        --base_lr 1e-3 \
        --min_lr 1e-6 \
        --layer_decay 0.75 \
        --weight_decay 0.05 \

# --------------- Finetune on ImageNet ---------------
# python mae_finetune.py \
#         --cuda \
#         --root path/to/imagenet_1k \
#         --dataset imagenet_1k \
#         -m vit_nano \
#         --batch_size 256 \
#         --img_size 224 \
#         --patch_size 16 \
#         --max_epoch 200 \
#         --wp_epoch 20 \
