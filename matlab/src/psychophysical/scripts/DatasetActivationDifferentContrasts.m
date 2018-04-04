function DatasetActivationDifferentContrasts(NetwrokName, DatasetName, outdir)

%% Network details
if strcmpi(NetwrokName, 'vgg16')
  net = vgg16;
elseif strcmpi(NetwrokName, 'vgg19')
  net = vgg19;
elseif strcmpi(NetwrokName, 'gogolenet')
  net = gogolenet;
elseif strcmpi(NetwrokName, 'inceptionv3')
  net = inceptionv3;
elseif strcmpi(NetwrokName, 'alexnet')
  net = alexnet;
end

%% Dataset details

% path of the dataset
if strcmpi(DatasetName, 'voc2012')
  DatasetPath = '/home/arash/Software/repositories/kernelphysiology/data/computervision/voc2012/JPEGImages/';
  ImageList = dir(sprintf('%s*.jpg', DatasetPath));
elseif strcmpi(DatasetName, 'ilsvrc2017')
  DatasetPath = '/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2017/Data/DET/test/';
  ImageList = dir(sprintf('%s*.JPEG', DatasetPath));
elseif strcmpi(DatasetName, 'ilsvrc-test')
  DatasetPath = '/home/ImageNet/Val_Images_RGB/';
  ImageList = dir(sprintf('%s*.png', DatasetPath));
end

NumImages = numel(ImageList);

outdir = sprintf('%s/%s/', outdir, DatasetName);

if ~exist(outdir, 'dir')
  mkdir(outdir);
end

%% Compute activation of kernels for different contrasts
SelectedImages = 1:NumImages;

layers = ConvInds(net, 5);

parfor i = SelectedImages
  inim = imread([DatasetPath, ImageList(i).name]);
  [~, ImageBaseName, ~] = fileparts(ImageList(i).name);
  ImageOutDir = sprintf('%s%s/', outdir, ImageBaseName);
  ActivationReport(i) = ActivationDifferentContrasts(net, inim, ImageOutDir, false, layers);
end

save([outdir, 'ActivationReport.mat'], 'ActivationReport');

%% Creating the matrix contrast versus accuracy

AverageKernelMatchings = zeros(NumImages, 6);

parfor i = SelectedImages
  fprintf('%s ', ImageList(i).name);
  AverageKernelMatchings(i, :) = ContrastVsAccuracy(ActivationReport(i));
end

save([outdir, 'AverageKernelMatchings.mat'], 'AverageKernelMatchings');

%% Printing the results
for i = [0:0.1:0.9, 0.999]
  meanvals = mean(AverageKernelMatchings(AverageKernelMatchings(:, 6) >= i, :));
  fprintf('>=%.2f %.2f %.2f %.2f %.2f %.2f\n', i, meanvals(1:5));
end

end