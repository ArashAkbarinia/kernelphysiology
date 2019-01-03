# fine tunning all the models in subfolders

# setting the GPU to be used
GPU_NUMBER=$1
export CUDA_VISIBLE_DEVICES=${GPU_NUMBER}

# the directory in which all subfolders are examined
EXPERIMENT_DIRECTORY=$2

#architecture
ARCHITECTURE=$3

# dataset
DATASET=$4

LR="1e-6"
LR_SCHEDULE="resnet"
DECAY="1e-6"
EPOCHS=50

SIZE_RANGE=0.1
NUM_AUGMENTATIONS=1
NOISE_AMOUNT=0.1

for dir in ${EXPERIMENT_DIRECTORY}/*/
do
    dir=${dir%*/}
    MODEL_PATH=${dir}/model_area_None.h5
    NAME=${dir##*/}
    echo ${NAME}
    python3 train_prominents.py ${DATASET} ${ARCHITECTURE} --gpus $GPU_NUMBER --workers 8 --load_weights ${MODEL_PATH} \
    --horizontal_flip --zoom_range ${SIZE_RANGE} --width_shift_range ${SIZE_RANGE} --height_shift_range ${SIZE_RANGE} --name ${NAME} \
    --illuminant_range 0.05 --local_illuminant_variation 0.03 \
    --contrast_range 0.05 -local_contrast_variation 0.03 \
    --gamma_range 0.3 3 \
    --gaussian_sigma 1.3 \
    --speckle_amount ${NOISE_AMOUNT} \
    --s_p_amount ${NOISE_AMOUNT} \
    --gaussian_amount ${NOISE_AMOUNT} \
    --poisson_noise \
    --optimiser adam --lr ${LR} --decay ${DECAY} --lr_schedule ${LR_SCHEDULE} --epochs ${EPOCHS} --num_augmentation ${NUM_AUGMENTATIONS}
done