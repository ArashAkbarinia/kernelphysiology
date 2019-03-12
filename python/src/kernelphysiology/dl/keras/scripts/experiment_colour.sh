DATASET=$1
NETWORKS=$2
GPU_NUMBER=$3

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 1 --chromaticity 0 0.1 0.25 0.50 --experiment_name reduce_chromaticity

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 1 --red_green 0 0.1 0.25 0.50 --experiment_name red_green

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 1 --yellow_blue 0 0.1 0.25 0.50 --experiment_name yellow_blue

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 1 --lightness 0 0.1 0.25 0.50 --experiment_name reduce_lightness

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 1 --invert_chromaticity --experiment_name invert_chromaticity

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 1 --invert_opponency --experiment_name invert_opponency

python3 predict_prominents.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers 1 --invert_lightness --experiment_name invert_lightness
