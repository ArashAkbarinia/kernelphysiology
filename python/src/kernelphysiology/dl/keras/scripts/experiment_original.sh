DATASET=$1
NETWORKS=$2
GPU_NUMBER=$3

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 1 --experiment_name original
