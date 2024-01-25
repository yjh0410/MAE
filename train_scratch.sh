# ------------------- Args setting -------------------
MODEL=$1
BATCH_SIZE=$2
DATASET=$3
DATASET_ROOT=$4
WORLD_SIZE=$5
RESUME=$6

# ------------------- Training setting -------------------
if [ $MODEL == "vit_huge" ]; then
    COLOR_FORMAT="rgb"
    MAX_EPOCH=200
    WP_EPOCH=20
    EVAL_EPOCH=10
    LAYER_DECAY=1.0
    DROP_PATH=0.2
    # Optimizer config
    OPTIMIZER="adamw"
    BASE_LR=1e-4
    MIN_LR=1e-6
    WEIGHT_DECAY=0.3
elif [ $MODEL == "vit_large" ]; then
    COLOR_FORMAT="rgb"
    MAX_EPOCH=200
    WP_EPOCH=20
    EVAL_EPOCH=10
    LAYER_DECAY=1.0
    DROP_PATH=0.2
    # Optimizer config
    OPTIMIZER="adamw"
    BASE_LR=1e-4
    MIN_LR=1e-6
    WEIGHT_DECAY=0.3
elif [ $MODEL == *"vit"* ]; then
    COLOR_FORMAT="rgb"
    MAX_EPOCH=300
    WP_EPOCH=20
    EVAL_EPOCH=10
    LAYER_DECAY=1.0
    DROP_PATH=0.1
    # Optimizer config
    OPTIMIZER="adamw"
    BASE_LR=1e-4
    MIN_LR=1e-6
    WEIGHT_DECAY=0.3
elif [ $MODEL == *"resnet"* ]; then
    COLOR_FORMAT="rgb"
    MAX_EPOCH=300
    WP_EPOCH=20
    EVAL_EPOCH=10
    LAYER_DECAY=1.0
    DROP_PATH=0.1
    # Optimizer config
    OPTIMIZER="adamw"
    BASE_LR=1e-4
    MIN_LR=1e-6
    WEIGHT_DECAY=0.05
elif [ $MODEL == *"rtcnet"* ]; then
    COLOR_FORMAT="bgr"
    MAX_EPOCH=300
    WP_EPOCH=20
    EVAL_EPOCH=10
    LAYER_DECAY=1.0
    DROP_PATH=0.1
    # Optimizer config
    OPTIMIZER="adamw"
    BASE_LR=1e-4
    MIN_LR=1e-6
    WEIGHT_DECAY=0.05
else
    COLOR_FORMAT="bgr"
    MAX_EPOCH=300
    WP_EPOCH=20
    EVAL_EPOCH=10
    LAYER_DECAY=1.0
    DROP_PATH=0.1
    # Optimizer config
    OPTIMIZER="adamw"
    BASE_LR=1e-4
    MIN_LR=1e-6
    WEIGHT_DECAY=0.05
fi


# ------------------- Dataset config -------------------
if [[ $DATASET == "cifar10" ]]; then
    IMG_SIZE=32
    PATCH_SIZE=2
    NUM_CLASSES=10
elif [[ $DATASET == "cifar100" ]]; then
    IMG_SIZE=32
    PATCH_SIZE=2
    NUM_CLASSES=100
elif [[ $DATASET == "imagenet_1k" || $DATASET == "imagenet_22k" ]]; then
    IMG_SIZE=224
    PATCH_SIZE=16
    NUM_CLASSES=1000
elif [[ $DATASET == "custom" ]]; then
    IMG_SIZE=224
    PATCH_SIZE=16
    NUM_CLASSES=2
else
    echo "Unknown dataset!!"
    exit 1
fi


# ------------------- Training pipeline -------------------
if [ $WORLD_SIZE == 1 ]; then
    python main_scratch.py \
            --cuda \
            --root ${DATASET_ROOT} \
            --dataset ${DATASET} \
            --color_format ${COLOR_FORMAT} \
            --model ${MODEL} \
            --resume ${RESUME} \
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
            --root ${DATASET_ROOT} \
            --dataset ${DATASET} \
            --color_format ${COLOR_FORMAT} \
            --model ${MODEL} \
            --resume ${RESUME} \
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