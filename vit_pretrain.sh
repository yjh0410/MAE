# Model config
MODEL="mae_vit_tiny"

# Batch size
BATCH_SIZE=256

# Epoch config
MAX_EPOCH=800
WP_EPOCH=40
EVAL_EPOCH=20

# Optimizer config
OPTIMIZER="adamw"
BASE_LR=0.00015
MIN_LR=0
WEIGHT_DECAY=0.05

# Mask ratio
MASK_RATIO=0.75

# Dataset config
DATASET="cifar10"
if [[ $DATASET == "cifar10" || $DATASET == "cifar100" ]]; then
    # Data root
    ROOT="none"
    # Image config
    IMG_SIZE=32
    PATCH_SIZE=2
elif [[ $DATASET == "imagenet_1k" || $DATASET == "imagenet_22k" ]]; then
    # Data root
    ROOT="path/to/imagenet"
    # Image config
    IMG_SIZE=224
    PATCH_SIZE=16
fi

# --------------- Pretrain ---------------
python mae_pretrain.py \
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
        --mask_ratio ${MASK_RATIO} \
