DATASET=$1
NETWORKS=$2
GPU_NUMBER=$3
WORKERS=$4
NAME_SUFFIX=$5

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --contrasts 0.05 0.10 0.15 0.30 0.50 --experiment_name contrast_reduction_$NAME_SUFFIX

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --gaussian_sigma 0.5 1.0 1.5 --experiment_name gaussian_blurring_$NAME_SUFFIX

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --gammas 0.1 0.3 3 10 --experiment_name gammas_$NAME_SUFFIX

