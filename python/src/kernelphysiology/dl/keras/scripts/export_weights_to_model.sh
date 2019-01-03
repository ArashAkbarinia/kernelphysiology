# generating models from the saved weights

# setting the GPU to be used
export CUDA_VISIBLE_DEVICES=$1

# the directory in which all subfolders are examined
experiment_directory=$2

#architecture
architecture=$3

# dataset
dataset=$4

for dir in $experiment_directory/*/
do
    dir=${dir%*/}
    weights_path=${dir}/model_area_None/model_weights_best.h5
    echo ${weights_path}
    model_path=${dir}/model_best.h5
    echo ${model_path}
    python3 export_weights_to_model.py ${weights_path} ${model_path} ${architecture} ${dataset}
done