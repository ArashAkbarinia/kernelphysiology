function DatasetActivationDifferentContrasts(DatasetName, outdir)

%% Network details
net = vgg16;

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

parfor i = SelectedImages
  inim = imread([DatasetPath, ImageList(i).name]);
  [~, ImageBaseName, ~] = fileparts(ImageList(i).name);
  ImageOutDir = sprintf('%s%s/', outdir, ImageBaseName);
  ActivationReport = ActivationDifferentContrasts(net, inim, ImageOutDir, false);
end

%%

AverageKernelMatchings = zeros(NumImages, 6);

parfor i = SelectedImages
  [~, ImageBaseName, ~] = fileparts(ImageList(i).name);
  ImageOutDir = sprintf('%s%s/', outdir, ImageBaseName);
  ActivationReport = load([ImageOutDir, 'ActivationReport.mat']);
  fprintf('%s ', ImageList(i).name);
  AverageKernelMatchings(i, :) = ContrastVsAccuracy(ActivationReport);
end

save('AverageKernelMatchings.mat', 'AverageKernelMatchings');

%%
for i = 0:0.1:1.0
  meanvals = mean(AverageKernelMatchings(AverageKernelMatchings(:, 6) >= i, :));
  fprintf('>=%.2f %.2f %.2f %.2f %.2f %.2f\n', i, meanvals(1:5));
end