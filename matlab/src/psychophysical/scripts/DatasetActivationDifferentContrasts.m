%% Network details
% net = vgg16;

%% Dataset details
DatasetName = 'ilsvrc2017';

% path of the dataset
if strcmpi(DatasetName, 'voc2012')
  DatasetPath = '/home/arash/Software/repositories/kernelphysiology/data/computervision/voc2012/JPEGImages/';
  ImageList = dir(sprintf('%s*.jpg', DatasetPath));
elseif strcmpi(DatasetName, 'ilsvrc2017')
  DatasetPath = '/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2017/Data/DET/test/';
  ImageList = dir(sprintf('%s*.JPEG', DatasetPath));
end

NumImages = numel(ImageList);

outdir = sprintf('/home/arash/Software/repositories/kernelphysiology/analysis/kernelsactivity/vgg16/%s/', DatasetName);

if ~exist(outdir, 'dir')
  mkdir(outdir);
end

%% Compute activation of kernels for different contrasts
SelectedImages = 401:600;

for i = SelectedImages
  inim = imread([DatasetPath, ImageList(i).name]);
  [~, ImageBaseName, ~] = fileparts(ImageList(i).name);
  ImageOutDir = sprintf('%s%s/', outdir, ImageBaseName);
  ActivationReport = ActivationDifferentContrasts(net, inim, ImageOutDir);
end

%%

for i = SelectedImages
  inim = imread([DatasetPath, ImageList(i).name]);
  [~, ImageBaseName, ~] = fileparts(ImageList(i).name);
  ImageOutDir = sprintf('%s%s/', outdir, ImageBaseName);
  ActivationReport = load([ImageOutDir, 'ActivationReport.mat']);
  fprintf('%s ', ImageList(i).name);
  AverageKernelMatching = ContrastVsAccuracy(ActivationReport);
end

%%
for i = 0:0.1:1.0
  meanvals = mean(all(all(:, 6) >= i, :));
  fprintf('>=%.2f %.2f %.2f %.2f %.2f %.2f\n', i, meanvals(1:5));
end