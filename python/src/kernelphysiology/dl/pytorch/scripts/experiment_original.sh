DATASET=$1
NETWORKS=$2
GPU_NUMBER=$3
WORKERS=$4

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --experiment_name original
