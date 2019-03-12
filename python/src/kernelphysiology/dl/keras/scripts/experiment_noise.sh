DATASET=$1
NETWORKS=$2
GPU_NUMBER=$3

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 4 --gaussian_sigma 0.5 1.0 1.5 --experiment_name gaussian_blurring

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 4 --s_p_noise 0.01 0.05 0.1 --experiment_name s_p_noise

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 4 --speckle_noise 0.01 0.05 0.1 --experiment_name speckle_noise

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 4 --gaussian_noise 0.01 0.05 0.1 --experiment_name gaussian_noise

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 4 --gammas 0.1 0.3 3 10 --experiment_name gammas

