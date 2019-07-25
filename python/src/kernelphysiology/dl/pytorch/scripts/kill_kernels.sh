NET_CONTRAST=$1
CHROMATICITY=$2
GPU_NUMBER=$3
LAYER_NAME=$4
NUM_KERNELS=$5
START_KERNEL=$6
START_KERNEL=${START_KERNEL:=0}
EXPERIMENT_NAME=$7
EXPERIMENT_NAME=${EXPERIMENT_NAME:="kill_convs"}
CONTRAST_VALUE=$8
CONTRAST_VALUE=${CONTRAST_VALUE:=0.3}


for (( i=START_KERNEL; i<$NUM_KERNELS; i++ ))
    do 
    echo "Killing Kernel index "$i;
    KERNEL_IND_PRINT=$(printf "%04d" $i)
    CURRENT_NAME="l_"$LAYER_NAME"_k_"$KERNEL_IND_PRINT
    > $EXPERIMENT_NAME".txt";
    echo $NET_CONTRAST","$CHROMATICITY","$CURRENT_NAME >> $EXPERIMENT_NAME".txt"
    FILE=$EXPERIMENT_NAME"/imagenet/contrast/"$CURRENT_NAME"/"$CURRENT_NAME"_contrast_"$CONTRAST_VALUE".csv"
    if [ ! -f "$FILE" ]; then
        echo "$FILE "
        python predict_image_classification.py imagenet $EXPERIMENT_NAME".txt" --experiment_name $EXPERIMENT_NAME --gpus $GPU_NUMBER -j 8 -b 512 --kill_kernels $LAYER_NAME $i --contrasts $CONTRAST_VALUE 
    fi

    FILE=$EXPERIMENT_NAME"/imagenet/original/"$CURRENT_NAME"/"$CURRENT_NAME"_original_1.csv"
    if [ ! -f "$FILE" ]; then
        echo "$FILE "
        python predict_image_classification.py imagenet $EXPERIMENT_NAME".txt" --experiment_name $EXPERIMENT_NAME --gpus $GPU_NUMBER -j 8 -b 512 --kill_kernels $LAYER_NAME $i
    fi
done
