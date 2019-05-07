DATASET=$1
NETWORKS=$2
GPU_NUMBER=$3
WORKERS=$4

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --chromaticity 0 0.1 0.25 0.50 --experiment_name ex_$5

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --red_green 0 0.1 0.25 0.50 --experiment_name ex_$5

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --yellow_blue 0 0.1 0.25 0.50 --experiment_name ex_$5

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --lightness 0 0.1 0.25 0.50 --experiment_name ex_$5

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --invert_chromaticity --experiment_name ex_$5

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --invert_opponency --experiment_name ex_$5

python3 predict_image_classification.py $DATASET $NETWORKS --gpus $GPU_NUMBER --workers $WORKERS --invert_lightness --experiment_name ex_$5
