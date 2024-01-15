# ------------------- Model setting -------------------
MODEL=$1 #"mae_resnet18"


# ------------------- Training setting -------------------
## Batch size
BATCH_SIZE=$2 #256

## Epoch
MAX_EPOCH=800
WP_EPOCH=40
EVAL_EPOCH=20

if [[ $MODEL == *"mae_vit"* ]]; then
    MASK_RATIO=0.75
    # Optimizer config
    OPTIMIZER="adamw"
    BASE_LR=0.00015
    MIN_LR=0
    WEIGHT_DECAY=0.05
    COLOR_FORMAT="rgb"
elif [[ $MODEL == *"mae_resnet"* ]]; then
    MASK_RATIO=0.75
    # Optimizer config
    OPTIMIZER="adamw"
    BASE_LR=0.00015
    MIN_LR=0
    WEIGHT_DECAY=0.05
    COLOR_FORMAT="rgb"
elif [[ $MODEL == *"mae_rtcnet"* ]]; then
    MASK_RATIO=0.75
    # Optimizer config
    OPTIMIZER="adamw"
    BASE_LR=0.00015
    MIN_LR=0
    WEIGHT_DECAY=0.05
    COLOR_FORMAT="bgr"
else
    echo "Unknown model!!"
    exit 1
fi


# ------------------- Dataset setting -------------------
DATASET="imagenet_1k"
if [[ $DATASET == "cifar10" ]]; then
    # Data root
    ROOT=$3 #"none"
    # Image config
    IMG_SIZE=32
    PATCH_SIZE=2
    NUM_CLASSES=10
elif [[ $DATASET == "cifar100" ]]; then
    # Data root
    ROOT=$3 #"none"
    # Image config
    IMG_SIZE=32
    PATCH_SIZE=2
    NUM_CLASSES=100
elif [[ $DATASET == "imagenet_1k" || $DATASET == "imagenet_22k" ]]; then
    # Data root
    ROOT=$3 #"/data/datasets/imagenet-1k/"
    # Image config
    IMG_SIZE=224
    PATCH_SIZE=16
    NUM_CLASSES=1000
elif [[ $DATASET == "custom" ]]; then
    # Data root
    ROOT=$3 #"/Users/liuhaoran/Desktop/python_work/classification/dataset/Animals/"
    # Image config
    IMG_SIZE=224
    PATCH_SIZE=16
    NUM_CLASSES=2
else
    echo "Unknown dataset!!"
    exit 1
fi


# ------------------- Training pipeline -------------------
WORLD_SIZE=$4
if [ $WORLD_SIZE == 1 ]; then
    python main_pretrain.py \
            --cuda \
            --root ${ROOT} \
            --dataset ${DATASET} \
            --color_format ${COLOR_FORMAT} \
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
            --mask_ratio ${MASK_RATIO}
elif [[ $WORLD_SIZE -gt 1 && $WORLD_SIZE -le 8 ]]; then
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port 1668 main_pretrain.py \
            --cuda \
            -dist \
            --root ${ROOT} \
            --dataset ${DATASET} \
            --color_format ${COLOR_FORMAT} \
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
            --mask_ratio ${MASK_RATIO}
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi