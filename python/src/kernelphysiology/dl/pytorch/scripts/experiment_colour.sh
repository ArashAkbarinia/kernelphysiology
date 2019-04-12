DATASET=$1
NETWORKS=$2
GPU_NUMBER=$3
WORKERS=$4
NAME_SUFFIX=$5

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --chromaticity 0 0.1 0.25 0.50 --experiment_name reduce_chromaticity_$NAME_SUFFIX

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --red_green 0 0.1 0.25 0.50 --experiment_name red_green_$NAME_SUFFIX

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --yellow_blue 0 0.1 0.25 0.50 --experiment_name yellow_blue_$NAME_SUFFIX

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --lightness 0 0.1 0.25 0.50 --experiment_name reduce_lightness_$NAME_SUFFIX

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --invert_chromaticity --experiment_name invert_chromaticity_$NAME_SUFFIX

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --invert_opponency --experiment_name invert_opponency_$NAME_SUFFIX

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --invert_lightness --experiment_name invert_lightness_$NAME_SUFFIX
