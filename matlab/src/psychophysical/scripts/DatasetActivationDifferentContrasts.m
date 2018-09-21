function DatasetActivationDifferentContrasts(NetwrokName, DatasetName, outdir, WhichLayers)

%% Network details
if isa(NetwrokName, 'SeriesNetwork') || isa(NetwrokName, 'DAGNetwork')
  net = NetwrokName;
elseif strcmpi(NetwrokName, 'vgg16')
  net = vgg16;
elseif strcmpi(NetwrokName, 'vgg19')
  net = vgg19;
elseif strcmpi(NetwrokName, 'vgg3c4x')
  net = load('vgg3c4x.mat');
  net = net.net;
elseif strcmpi(NetwrokName, 'vgg5x')
  net = load('vgg5x.mat');
  net = net.net;
elseif strcmpi(NetwrokName, 'googlenet')
  net = googlenet;
elseif strcmpi(NetwrokName, 'inceptionv3')
  net = inceptionv3;
elseif strcmpi(NetwrokName, 'alexnet')
  net = alexnet;
elseif strcmpi(NetwrokName, 'resnet50')
  net = resnet50;
elseif strcmpi(NetwrokName, 'resnet101')
  net = resnet101;
elseif strcmpi(NetwrokName, 'inceptionresnetv2')
  net = inceptionresnetv2;
elseif strcmpi(NetwrokName, 'squeezenet')
  net = squeezenet;
end

%% Dataset details

imdb = [];

% path of the dataset
if strcmpi(DatasetName, 'voc2012')
  DatasetPath = '/home/arash/Software/repositories/kernelphysiology/data/computervision/voc2012/JPEGImages/';
  ImageList = dir(sprintf('%s*.jpg', DatasetPath));
elseif strcmpi(DatasetName, 'ilsvrc2017')
  DatasetPath = '/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2017/Data/DET/test/';
  ImageList = dir(sprintf('%s*.JPEG', DatasetPath));
elseif strcmpi(DatasetName, 'ilsvrc-test')
  DatasetPath = '/home/ImageNet/Val_Images_RGB/';
%   DatasetPath = '/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2012/Val_Images_RGB/';
  ImageList = dir(sprintf('%s*.png', DatasetPath));
elseif strcmpi(DatasetName, 'cifar10')
  imdb = load('/home/arash/Software/repositories/kernelphysiology/matlab/data/datasets/cifar/cifar10/imdb-org.mat');
elseif strcmpi(DatasetName, 'cifar100')
  imdb = load('/home/arash/Software/repositories/kernelphysiology/matlab/data/datasets/cifar/cifar100/imdb-org.mat');
end

outdir = sprintf('%s/%s/', outdir, DatasetName);

if ~exist(outdir, 'dir')
  mkdir(outdir);
end

%% Compute activation of kernels for different contrasts

if nargin < 4
  WhichLayers = 'max';
end

if strcmpi(WhichLayers, 'conv')
  layers = ConvInds(net, inf);
elseif strcmpi(WhichLayers, 'max')
  layers = BeforeMaxInds(net, 5);
elseif strcmpi(WhichLayers, 'corr')
  layers = MaxInds(net, inf);
  layers = (2:min(layers(end), 11))';
end

ActivitiesDir = sprintf('%s/%s/activities/', outdir, WhichLayers);
if ~exist(ActivitiesDir, 'dir')
  mkdir(ActivitiesDir);
end
ReportDir = sprintf('%s/%s/reports/', outdir, WhichLayers);
if ~exist(ReportDir, 'dir')
  mkdir(ReportDir);
end

ContrastLevels = [5, 15, 50, 100];
nTopXPreds = 5;

if ~isempty(imdb)
  TestImages = uint8(imdb.images.data(:, :, :, imdb.images.set == 3));
  ActivationReport = PerformWithImdb(net, TestImages, layers, ActivitiesDir, WhichLayers, ContrastLevels, nTopXPreds); %#ok
else
  ActivationReport = PerformWithImageList(net, ImageList, layers, ActivitiesDir, WhichLayers, ContrastLevels, nTopXPreds); %#ok
end

save([ReportDir, 'ActivationReport.mat'], 'ActivationReport');

end

function ActivationReport = PerformWithImdb(net, TestImages, layers, ActivitiesDir, WhichLayers, ContrastLevels, nTopXPreds)

NumImages = size(TestImages, 4);
data = cell(NumImages);

parfor i = 1:NumImages
  inim = TestImages(:, :, :, i);
  ImageBaseName = sprintf('im%.6i', i);
  ImageOutDir = sprintf('%s/%s/', ActivitiesDir, ImageBaseName);
  data{i} = PerformOneImage(net, inim, ImageOutDir, layers, ContrastLevels, WhichLayers, nTopXPreds);
end

ActivationReport.data = data;
ActivationReport.info.nImages = NumImages;
ActivationReport.info.nContrasts = numel(ContrastLevels);
ActivationReport.info.nLayers = size(layers, 1);
ActivationReport.info.nTopXPreds = nTopXPreds;

end

function ActivationReport = PerformWithImageList(net, ImageList, layers, ActivitiesDir, WhichLayers, ContrastLevels, nTopXPreds)

NumImages = numel(ImageList);
data = cell(NumImages, 1);

parfor i = 1:NumImages
  inim = imread([ImageList(i).folder, '/', ImageList(i).name]);
  [~, ImageBaseName, ~] = fileparts(ImageList(i).name);
  ImageOutDir = sprintf('%s/%s/', ActivitiesDir, ImageBaseName);
  data{i} = PerformOneImage(net, inim, ImageOutDir, layers, ContrastLevels, WhichLayers, nTopXPreds);
end

ActivationReport.data = data;
ActivationReport.info.nImages = NumImages;
ActivationReport.info.nContrasts = numel(ContrastLevels);
ActivationReport.info.nLayers = size(layers, 1);
ActivationReport.info.nTopXPreds = nTopXPreds;

end

function data = PerformOneImage(net, inim, ImageOutDir, layers, ContrastLevels, WhichLayers, nTopXPreds)

if strcmpi(WhichLayers, 'conv')
  data = ActivationDifferentContrasts(net, inim, ImageOutDir, false, layers, ContrastLevels);
elseif strcmpi(WhichLayers, 'max')
  data = MaxActivationDifferentContrasts(net, inim, ImageOutDir, false, layers, ContrastLevels);
elseif strcmpi(WhichLayers, 'corr')
  data = ActivationCorrDifferentContrasts(net, inim, ImageOutDir, false, layers, ContrastLevels, nTopXPreds);
end

end
