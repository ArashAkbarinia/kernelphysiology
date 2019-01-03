python3 evaluate_prominents.py stl10 stl10.txt --preprocessing resnet20  --gpus 0 --workers 8 --name ExpSalt --s_p_noise 0.01 0.05 0.1
python3 evaluate_prominents.py stl10 stl10.txt --preprocessing resnet20  --gpus 0 --workers 8 --name ExpSpeckle --speckle_noise 0.01 0.05 0.1
python3 evaluate_prominents.py stl10 stl10.txt --preprocessing resnet20  --gpus 0 --workers 8 --name ExpGaussian --gaussian_noise 0.01 0.05 0.1
python3 evaluate_prominents.py stl10 stl10.txt --preprocessing resnet20  --gpus 0 --workers 8 --name ExpGamma --gammas 0.1 0.3 0.5 0.8 1.2 2.0
python3 evaluate_prominents.py stl10 stl10.txt --preprocessing resnet20  --gpus 0 --workers 8 --name ExpContrast --contrasts 0.01 0.05 0.15 0.30 0.50 0.75 1.0
python3 evaluate_prominents.py stl10 stl10.txt --preprocessing resnet20  --gpus 0 --workers 8 --name ExpIlluminants --illuminants 0.05 0.25 0.50 0.75
python3 evaluate_prominents.py stl10 stl10.txt --preprocessing resnet20  --gpus 0 --workers 8 --name ExpBlur --gaussian_sigma 0.1 0.3 0.5 0.7 0.9
python3 evaluate_prominents.py stl10 stl10.txt --preprocessing resnet20  --gpus 0 --workers 8 --name ExpPoisson --poisson_noise
