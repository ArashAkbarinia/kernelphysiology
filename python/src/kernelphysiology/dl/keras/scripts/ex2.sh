dataset_names=("cifar10" "cifar100" "stl10")
#dataset_names=("cifar10")
GPU_NUMBER=2
#noise_amount=("0.05" "0.1" "0.2")
noise_amount=("0.2")
gaussian_sigmas=("2")
#num_augmentations=("0" "1" "2" "3")
num_augmentations=("0" "3")
#lrs=("1e-2" "3.3e-3" "6.6e-3" "1e-3")
lrs=("5e-3")
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
          for (( g=0; g<${#gaussian_sigmas[@]}; g++))
          do
            name="gpu_arash_${GPU_NUMBER}_i05l03_c05l03_s01_g0330_b${gaussian_sigmas[g]}_n${noise_amount[n]}_t${num_augmentations[a]}_ol${lrs[l]}_od${decays[d]}"
            python3 train_prominents.py ${dataset_names[f]} resnet50 --gpus $GPU_NUMBER --workers 8 --horizontal_flip --zoom_range 0.1 --width_shift_range 0.1 --height_shift_range 0.1 --name $name --illuminant_range 0.05 --contrast_range 0.05 --local_illuminant_variation 0.03 --local_contrast_variation 0.03 --gamma_range 0.3 3 --gaussian_sigma ${gaussian_sigmas[g]} --speckle_amount ${noise_amount[n]} --s_p_amount ${noise_amount[n]} --gaussian_amount ${noise_amount[n]} --poisson_noise --optimiser adam --lr ${lrs[l]} --decay ${decays[d]} --lr_schedule arash --epochs 200 --num_augmentation ${num_augmentations[a]}
            name="gpu_arash_${GPU_NUMBER}_i05l03_c05l03_s01_g0330_b${gaussian_sigmas[g]}_n${noise_amount[n]}_t${num_augmentations[a]}_ol${lrs[l]}_od${decays[d]}_dog"
            python3 train_prominents.py ${dataset_names[f]} resnet50 --gpus $GPU_NUMBER --workers 8 --horizontal_flip --zoom_range 0.1 --width_shift_range 0.1 --height_shift_range 0.1 --name $name --illuminant_range 0.05 --contrast_range 0.05 --local_illuminant_variation 0.03 --local_contrast_variation 0.03 --gamma_range 0.3 3 --gaussian_sigma ${gaussian_sigmas[g]} --speckle_amount ${noise_amount[n]} --s_p_amount ${noise_amount[n]} --gaussian_amount ${noise_amount[n]} --poisson_noise --optimiser adam --lr ${lrs[l]} --decay ${decays[d]} --lr_schedule arash --epochs 200 --num_augmentation ${num_augmentations[a]} --initialise dog
          done
        done
      done
    done
  done
done

