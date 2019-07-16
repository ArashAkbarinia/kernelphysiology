NET_CONTRAST=$1
CHROMATICITY=$2
GPU_NUMBER=$3
LAYER_NAME=$4
NUM_KERNELS=$5
START_KERNEL=$6
START_KERNEL=${START_KERNEL:=0}
EXPERIMENT_NAME=$7
EXPERIMENT_NAME=${EXPERIMENT_NAME:="kill_convs"}


for (( i=START_KERNEL; i<$NUM_KERNELS; i++ ))
    do 
    echo "Killing Kernel index "$i;
    KERNEL_IND_PRINT=$(printf "%04d" $i)
    CURRENT_NAME="l_"$LAYER_NAME"_k_"$KERNEL_IND_PRINT
    > $EXPERIMENT_NAME"_"$GPU_NUMBER".txt";
    echo $NET_CONTRAST","$CHROMATICITY","$CURRENT_NAME >> $EXPERIMENT_NAME"_"$GPU_NUMBER".txt"
    python predict_image_classification.py imagenet $EXPERIMENT_NAME"_"$GPU_NUMBER".txt" --experiment_name $EXPERIMENT_NAME --gpus $GPU_NUMBER -j 8 -b 512 --kill_kernels $LAYER_NAME $i --contrasts 0.3 
    python predict_image_classification.py imagenet $EXPERIMENT_NAME"_"$GPU_NUMBER".txt" --experiment_name $EXPERIMENT_NAME --gpus $GPU_NUMBER -j 8 -b 512 --kill_kernels $LAYER_NAME $i
done
