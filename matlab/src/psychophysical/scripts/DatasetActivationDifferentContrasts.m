function DatasetActivationDifferentContrasts(NetwrokName, DatasetName, outdir)

%% Network details
if strcmpi(NetwrokName, 'vgg16')
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

AverageKernelMatchingsEqTop = zeros(NumImages, 6);

parfor i = SelectedImages
  AverageKernelMatchingsEqTop(i, :) = ContrastVsAccuracy(ActivationReport(i), false);
end

save([outdir, 'AverageKernelMatchingsEqTop.mat'], 'AverageKernelMatchingsEqTop');

%% Printing the results
fprintf('All results\n');
for i = [0:0.1:0.9, 0.999]
  meanvals = mean(AverageKernelMatchingsEqTop(AverageKernelMatchingsEqTop(:, 6) >= i, :));
  fprintf('>=%.2f %.2f %.2f %.2f %.2f %.2f\n', i, meanvals(1:5));
end

%% if GT exist
if false %strcmpi(DatasetName, 'ilsvrc-test')
  ValidationDir = '/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2012/validation/';
  
  labels = dlmread(sprintf('%sILSVRC2012_validation_ground_truth.txt', ValidationDir));
  ImageInfos = load(sprintf('%sILSVRC2012_validation_meta.mat', ValidationDir));
  
  %% comuting whether the network has been correct or not
  AverageKernelMatchingsAll = zeros(NumImages, 7);
  
  parfor i = SelectedImages
    AverageKernelMatchingsTmp = ContrastVsAccuracy(ActivationReport(i));
    
    % right now just for the last contrast
    ContrastNames = fieldnames(ActivationReport(i).cls);
    nContrasts = numel(ContrastNames);
    
    for j = nContrasts
      prediction = ActivationReport(i).cls.(ContrastNames{j}).prediction.type;
      AcceptedResults = strsplit(ImageInfos.synsets(labels(i)).words, ', ');
      MatchedAny = find(strcmpi(AcceptedResults, prediction), 1);
      AverageKernelMatchingsAll(i, :) = [AverageKernelMatchingsTmp, ~isempty(MatchedAny)];
    end
  end
  
  AverageKernelMatchingsEqTop(:, 7) = AverageKernelMatchingsAll(:, 7);
  
  save([outdir, 'AverageKernelMatchingsAll.mat'], 'AverageKernelMatchingsAll');
  save([outdir, 'AverageKernelMatchingsEqTop.mat'], 'AverageKernelMatchingsEqTop');
  
  %% printing the result according to being correct or not
  fprintf('Printing for all\n');
  PrintAverageKernelMatchings(AverageKernelMatchingsAll);
  fprintf('Printing for top\n');
  PrintAverageKernelMatchings(AverageKernelMatchingsEqTop);
end

end

function PrintAverageKernelMatchings(AverageKernelMatchings, SpecificRange)

if nargin < 2
  SpecificRange = true;
end

%%
for j = [0, 1, 2]
  switch j
    case 0
      fprintf('Network being incorrect\n');
      WhichResults = AverageKernelMatchings(:, 7) == j;
    case 1
      fprintf('Network being correct\n');
      WhichResults = AverageKernelMatchings(:, 7) == j;
    case 2
      WhichResults = true(size(AverageKernelMatchings(:, 7)));
      fprintf('All results\n');
  end
  if SpecificRange
    for i = 0:0.1:0.9
      if i ~= 0.90
        UpperBound = i + 0.1;
      else
        UpperBound = 1.0001;
      end
      meanvals = mean(AverageKernelMatchings(AverageKernelMatchings(:, 6) >= i & AverageKernelMatchings(:, 6) < UpperBound & WhichResults, :));
      fprintf('>=%.2f<%.2f %.2f %.2f %.2f %.2f %.2f\n', i, UpperBound, meanvals(1:5));
    end
  else
    for i = [0:0.1:0.9, 0.999]
      meanvals = mean(AverageKernelMatchings(AverageKernelMatchings(:, 6) >= i & WhichResults, :));
      fprintf('>=%.2f %.2f %.2f %.2f %.2f %.2f\n', i, meanvals(1:5));
    end
  end
end

end
