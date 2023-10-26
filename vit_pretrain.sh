# --------------- Pretrain on CIFAR ---------------
# python mae_pretrain.py \
#         --cuda \
#         --dataset cifar10 \
#         -m mae_vit_tiny \
#         --batch_size 256 \
#         --img_size 32 \
#         --patch_size 2 \
#         --max_epoch 400 \
#         --wp_epoch 40 \
#         --eval_epoch 20 \


# --------------- Pretrain on ImageNet ---------------
python mae_pretrain.py \
        --cuda \
        --root /data/datasets/imagenet-1k \
        --dataset imagenet_1k \
        -m mae_vit_tiny \
        --batch_size 256 \
        --img_size 224 \
        --patch_size 16 \
        --max_epoch 800 \
        --wp_epoch 40 \
        --optimizer adamw \
        --base_lr 0.00015 \
        --weight_decay 0.05 \
        --mask_ratio 0.75 \
