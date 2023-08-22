# --------------- Pretrain on CIFAR ---------------
python mae_pretrain.py \
        --dataset cifar10 \
        -m mae_vit_nano \
        --batch_size 256 \
        --img_size 32 \
        --patch_size 2 \
        --max_epoch 400 \
        --wp_epoch 40


# --------------- Pretrain on ImageNet ---------------
# python mae_pretrain.py \
#         --dataset imagenet_1k \
#         -m mae_vit_nano \
#         --batch_size 256 \
#         --img_size 224 \
#         --patch_size 16 \
#         --max_epoch 400 \
#         --wp_epoch 40
