% extracting and printing accuracy and each contrast level

NetNames = {'vgg16', 'vgg19', 'alexnet', 'googlenet', 'inceptionv3', 'resnet50', 'resnet101', 'vgg3c4x'};

DatasetName = 'ilsvrc-test';
AnalysisDir = '/mnt/sdb/Dropbox/Jobs/Gießen/KarlArash/dnn/analysis/kernelsactivity/';

NumNets = numel(NetNames);

% in the first experiment we tried it with the first five convolutional
% layers
AccuracuReports = zeros(5, NumNets);

for i = 1:3
  NetwrokName = NetNames{i};
  
  if strcmpi(NetwrokName, 'vgg16')
    net = vgg16;
  elseif strcmpi(NetwrokName, 'vgg19')
    net = vgg19;
  elseif strcmpi(NetwrokName, 'vgg3c4x')
    net = load('/mnt/sdb/Dropbox/Jobs/Gießen/nets/vgg3c4x.mat');
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
  
  layers = ConvInds(net, inf);
  j = 1;
  a = [];
  for l = layers
    a(:, :, j) = net.Layers(l).NumFilters(1);
    j = j + 1;
  end
  
  load([AnalysisDir, NetwrokName, '/ilsvrc-test/hist/PairwiseReport.mat']);
  maxvals = PairwiseReport.avgs.max;
  arash = maxvals ./ (1./repmat(a, [1, 10, 1]));
  
  %%
  [~, nComparisons, nLayers] = size(arash);
  
  idontknow.(NetwrokName) = [];
  for n = 1:nComparisons
    NonNaN = ~isnan(arash(:, n, 1));
    meanvals = mean(arash(NonNaN, n, :));
    idontknow.(NetwrokName)(n, :) = permute(meanvals, [3, 1, 2]);
  end
%   idontknow.(NetwrokName) = idontknow.(NetwrokName) ./ max(idontknow.(NetwrokName), [], 1);
  
  %%
  
  
%   new.(NetwrokName) = old.(NetwrokName) ./ (1./repmat(a, [10, 1]));
%   new.(NetwrokName) = new.(NetwrokName) ./ max(new.(NetwrokName), [], 1);
  
%   outdir = [AnalysisDir, NetwrokName, '/'];
%   
%   if exist(sprintf('%s/%s/AccuracyReport.mat', outdir, DatasetName), 'file')
%     MatPath = sprintf('%s/%s/AccuracyReport.mat', outdir, DatasetName);
%     AccuracyReport = load(MatPath);
%   else
%     MatPath = sprintf('%s/%s/ActivationReport.mat', outdir, DatasetName);
%     ActivationReport = load(MatPath, 'ActivationReport');
%     AccuracyReport = ReportAccuracyAtContrastLevel(ActivationReport.ActivationReport, DatasetName);
%     
%     save(sprintf('%s/%s/AccuracyReport.mat', outdir, DatasetName), '-struct', 'AccuracyReport');
%   end
%   
%   FormatedString = [repmat('%.2f ', [1, size(AccuracyReport.ContrastAccuracyMatrix, 2)]), '\n'];
%   fprintf(FormatedString, permute(mean(AccuracyReport.ContrastAccuracyMatrix, 1), [2, 1]));
%   AccuracuReports(:, i) = permute(mean(AccuracyReport.ContrastAccuracyMatrix, 1), [2, 1]);
end
