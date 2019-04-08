DATASET=$1
NETWORKS=$2
GPU_NUMBER=$3
WORKERS=$4

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --s_p_noise 0.01 0.05 0.1 --experiment_name s_p_noise

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --speckle_noise 0.01 0.05 0.1 --experiment_name speckle_noise

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --gaussian_noise 0.01 0.05 0.1 --experiment_name gaussian_noise

