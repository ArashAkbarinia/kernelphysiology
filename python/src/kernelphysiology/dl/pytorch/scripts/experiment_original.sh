DATASET=$1
NETWORKS=$2
GPU_NUMBER=$3
WORKERS=$4

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --experiment_name ex_$5

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --original_rgb --experiment_name ex_$5

