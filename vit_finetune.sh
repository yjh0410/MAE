# Model config
MODEL="mae_vit_tiny"
MAE_PRETRAINED_MODEL="weights/cifar10/mae_vit_tiny/checkpoint-0.pth"

# Batch size
BATCH_SIZE=256

# Dataset config
DATASET="cifar10"
if [[ $DATASET == "cifar10" || $DATASET == "cifar100" ]]; then
    # Data root
    ROOT="none"
    # Image config
    IMG_SIZE=32
    PATCH_SIZE=2
    # Epoch config
    MAX_EPOCH=400
    WP_EPOCH=40
    EVAL_EPOCH=20
    # Optimizer config
    OPTIMIZER="adamw"
    BASE_LR=1e-3
    MIN_LR=1e-6
    WEIGHT_DECAY=0.05
    LAYER_DECAY=0.65
elif [[ $DATASET == "imagenet_1k" || $DATASET == "imagenet_22k" ]]; then
    # Data root
    ROOT="path/to/imagenet"
    # Image config
    IMG_SIZE=224
    PATCH_SIZE=16
    # Epoch config
    MAX_EPOCH=800
    WP_EPOCH=40
    EVAL_EPOCH=20
    # Optimizer config
    OPTIMIZER="adamw"
    BASE_LR=0.0005
    MIN_LR=1e-6
    WEIGHT_DECAY=0.05
    LAYER_DECAY=0.65
fi


# --------------- Finetune on CIFAR ---------------
python mae_finetune.py \
        --cuda \
        --root ${ROOT} \
        --dataset ${DATASET} \
        -m ${MODEL} \
        --batch_size ${BATCH_SIZE} \
        --img_size ${IMG_SIZE} \
        --patch_size ${PATCH_SIZE} \
        --max_epoch ${MAX_EPOCH} \
        --wp_epoch ${WP_EPOCH} \
        --eval_epoch ${EVAL_EPOCH} \
        --optimizer ${OPTIMIZER} \
        --base_lr ${BASE_LR} \
        --min_lr ${MIN_LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --reprob 0.25 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --mae_pretrained ${MAE_PRETRAINED_MODEL} \
