# Note that the configuration in this script accessory are not fully aligned with the official configuration.

# Model config
MODEL="vit_tiny"
if [ $MODEL == "vit_huge" ]; then
    MAX_EPOCH=200
    WP_EPOCH=20
    EVAL_EPOCH=10
    LAYER_DECAY=1.0
    DROP_PATH=0.2
elif [ $MODEL == "vit_large" ]; then
    MAX_EPOCH=200
    WP_EPOCH=20
    EVAL_EPOCH=10
    LAYER_DECAY=1.0
    DROP_PATH=0.2
else
    MAX_EPOCH=300
    WP_EPOCH=20
    EVAL_EPOCH=10
    LAYER_DECAY=1.0
    DROP_PATH=0.1
fi

# Batch size
BATCH_SIZE=256

# Optimizer config
OPTIMIZER="adamw"
BASE_LR=0.0001
MIN_LR=1e-6
WEIGHT_DECAY=0.3

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
else
    echo "Unknown dataset!!"
    exit 1
fi

# ------------------- Training pipeline -------------------
WORLD_SIZE=1
if [ $WORLD_SIZE == 1 ]; then
    python main_scratch.py \
            --cuda \
            --root ${ROOT} \
            --dataset ${DATASET} \
            -m ${MODEL} \
            --batch_size ${BATCH_SIZE} \
            --img_size ${IMG_SIZE} \
            --patch_size ${PATCH_SIZE} \
            --drop_path ${DROP_PATH} \
            --max_epoch ${MAX_EPOCH} \
            --wp_epoch ${WP_EPOCH} \
            --eval_epoch ${EVAL_EPOCH} \
            --optimizer ${OPTIMIZER} \
            --base_lr ${BASE_LR} \
            --min_lr ${MIN_LR} \
            --layer_decay ${LAYER_DECAY} \
            --weight_decay ${WEIGHT_DECAY} \
            --reprob 0.25 \
            --mixup 0.8 \
            --cutmix 1.0
elif [[ $WORLD_SIZE -gt 1 && $WORLD_SIZE -le 8 ]]; then
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port 1668 main_scratch.py \
            --cuda \
            -dist \
            --root ${ROOT} \
            --dataset ${DATASET} \
            -m ${MODEL} \
            --batch_size ${BATCH_SIZE} \
            --img_size ${IMG_SIZE} \
            --patch_size ${PATCH_SIZE} \
            --drop_path ${DROP_PATH} \
            --max_epoch ${MAX_EPOCH} \
            --wp_epoch ${WP_EPOCH} \
            --eval_epoch ${EVAL_EPOCH} \
            --optimizer ${OPTIMIZER} \
            --base_lr ${BASE_LR} \
            --min_lr ${MIN_LR} \
            --layer_decay ${LAYER_DECAY} \
            --weight_decay ${WEIGHT_DECAY} \
            --reprob 0.25 \
            --mixup 0.8 \
            --cutmix 1.0
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi