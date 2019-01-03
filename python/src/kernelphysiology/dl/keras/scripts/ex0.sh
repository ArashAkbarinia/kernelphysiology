dataset_names=("cifar10" "cifar100" "stl10")
#dataset_names=("cifar100")
GPU_NUMBER=0
noise_amount=("0.05" "0.1" "0.2" "0.3")
num_augmentations=("1" "2")
lrs=("3.3e-3" "5e-3" "6.6e-3" "1e-3")
#lrs=("5e-3")
#decays=("1e-5" "1e-6" "1e-7")
decays=("1e-6")

for (( f=0; f<${#dataset_names[@]}; f++))
do
  for (( n=0; n<${#noise_amount[@]}; n++))
  do
    for (( a=0; a<${#num_augmentations[@]}; a++))
    do
      for (( l=0; l<${#lrs[@]}; l++))
      do
        for (( d=0; d<${#decays[@]}; d++))
        do
          name="g1_gpu_${GPU_NUMBER}_i05l03_c05l03_s01_g0330_b13_n${noise_amount[n]}_t${num_augmentations[a]}_ol${lrs[l]}_od${decays[d]}"
          python3 train_prominents.py ${dataset_names[f]} resnet20 --gpus $GPU_NUMBER --workers 8 --horizontal_flip --zoom_range 0.1 --width_shift_range 0.1 --height_shift_range 0.1 --name $name --illuminant_range 0.05 --contrast_range 0.05 --local_illuminant_variation 0.03 --local_contrast_variation 0.03 --gamma_range 0.3 3 --gaussian_sigma 1.3 --speckle_amount ${noise_amount[n]} --s_p_amount ${noise_amount[n]} --gaussian_amount ${noise_amount[n]} --poisson_noise --optimiser adam --lr ${lrs[l]} --decay ${decays[d]} --lr_schedule resnet --epochs 200 --num_augmentation ${num_augmentations[a]} --initialise all --same_channels
        done
      done
    done
  done
done

