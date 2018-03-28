%% Network details
net = vgg16;

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
NumAnalysedImages = 2;

% SelectedImages = randi(NumImages, [1, NumAnalysedImages]);
SelectedImages = 100:136;

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
  AverageKernelMatching = ContrastVsAccuracy(ActivationReport);
end
