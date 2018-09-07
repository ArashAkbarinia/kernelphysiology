function [AverageKernelMatchingsSameOut, AverageKernelMatchingsAllOut] = AnalyseActivationReport(ActivationReportPath, DatasetName)
%AnalyseActivationReport Summary of this function goes here
%   Detailed explanation goes here

% for now we only have groundtruth for ilsvrc-test and cifars

imdb  = [];

if strcmpi(DatasetName, 'ilsvrc-test')
  ValidationDir = '/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2012/validation/';
  
  labels = dlmread(sprintf('%sILSVRC2012_validation_ground_truth.txt', ValidationDir));
  ImageInfos = load(sprintf('%sILSVRC2012_validation_meta.mat', ValidationDir));
elseif strcmpi(DatasetName, 'cifar10')
  imdb = load('/home/arash/Software/repositories/kernelphysiology/matlab/data/datasets/cifar/cifar10/imdb-org.mat');
elseif strcmpi(DatasetName, 'cifar100')
  imdb = load('/home/arash/Software/repositories/kernelphysiology/matlab/data/datasets/cifar/cifar100/imdb-org.mat');
else
  error('No groundtruth is available for dataset: %s\n', DatasetName);
end

outdir = fileparts(ActivationReportPath);

%% Creating the matrix contrast versus accuracy

AverageKernelMatchingsSameOutPath = [outdir, '/AverageKernelMatchingsSameOut.mat'];
AverageKernelMatchingsAllOutPath = [outdir, '/AverageKernelMatchingsAllOut.mat'];
if ~exist(AverageKernelMatchingsSameOutPath, 'file')
  ActivationReport = load(ActivationReportPath);
  ActivationReport = ActivationReport.ActivationReport;
  
  NumImages = numel(ActivationReport);
  
  if ~isempty(imdb)
    GroundTruths = imdb.images.labels(imdb.images.set == 3);
    %     GroundTruths =  categorical(imdb.meta.classes(imdb.images.labels(imdb.images.set == 3)));
  else
    TestLabels = ImageInfos.synsets;
    
    GroundTruths = cell(NumImages, 1);
    for i = 1:NumImages
      GroundTruths{i} = TestLabels(labels(i)).words;
    end
  end
  
  [nContrasts, ~, NumLayers] = size(ActivationReport{1}.CompMatrix);
  
  SameOutPixelsTop = zeros(NumImages, NumLayers);
  AllOutPixelsTop = zeros(NumImages, NumLayers);
  SameOutPixelsHist = zeros(NumImages, NumLayers);
  AllOutPixelsHist = zeros(NumImages, NumLayers);
  SameOutKernelsHist = zeros(NumImages, NumLayers);
  AllOutKernelsHist = zeros(NumImages, NumLayers);
  
  predictions = cell(NumImages, 1);
  corrects = zeros(NumImages, nContrasts);
  scores = zeros(NumImages, nContrasts);
  % TODO: maybe get the STDs as well.
  for i = 1:NumImages
    SameOutCompare = ContrastVsAccuracy(ActivationReport{i}, false);
    SameOutPixelsTop(i, :) = SameOutCompare.pixels.top.avg;
    SameOutPixelsHist(i, :) = SameOutCompare.pixels.hist.avg;
    SameOutKernelsHist(i, :) = SameOutCompare.kernels.hist.avg;
    predictions{i} = SameOutCompare.predictions;
    
    AllOutCompare = ContrastVsAccuracy(ActivationReport{i}, true);
    AllOutPixelsTop(i, :) = AllOutCompare.pixels.top.avg;
    AllOutPixelsHist(i, :) = AllOutCompare.pixels.hist.avg;
    AllOutKernelsHist(i, :) = AllOutCompare.kernels.hist.avg;
    
    if ~isempty(imdb)
      MatchedAny = CheckCifar(SameOutCompare, GroundTruths(i));
    else
      MatchedAny = CheckIlsvrc(SameOutCompare, GroundTruths{i});
    end
    
    corrects(i, :) = MatchedAny';
    scores(i, :) = cell2mat(SameOutCompare.predictions(:, 2))';
  end
  
  % for those equal to top contrast
  AverageKernelMatchingsSameOut.pixels.top.avg = SameOutPixelsTop;
  AverageKernelMatchingsSameOut.pixels.hist.avg = SameOutPixelsHist;
  AverageKernelMatchingsSameOut.kernels.hist.avg = SameOutKernelsHist;
  AverageKernelMatchingsSameOut.predictions = predictions;
  AverageKernelMatchingsSameOut.corrects = corrects;
  AverageKernelMatchingsSameOut.scores = scores;
  
  % for all
  AverageKernelMatchingsAllOut.pixels.top.avg = AllOutPixelsTop;
  AverageKernelMatchingsAllOut.pixels.hist.avg = AllOutPixelsHist;
  AverageKernelMatchingsAllOut.kernels.hist.avg = AllOutKernelsHist;
  AverageKernelMatchingsAllOut.predictions = predictions;
  AverageKernelMatchingsAllOut.corrects = corrects;
  AverageKernelMatchingsAllOut.scores = scores;
  
  save(AverageKernelMatchingsSameOutPath, 'AverageKernelMatchingsSameOut');
  save(AverageKernelMatchingsAllOutPath, 'AverageKernelMatchingsAllOut');
else
  AverageKernelMatchingsSameOut = load(AverageKernelMatchingsSameOutPath);
  AverageKernelMatchingsSameOut = AverageKernelMatchingsSameOut.AverageKernelMatchingsSameOut;
  
  AverageKernelMatchingsAllOut = load(AverageKernelMatchingsAllOutPath);
  AverageKernelMatchingsAllOut = AverageKernelMatchingsAllOut.AverageKernelMatchingsAllOut;
end

%% printing the result according to being correct or not
fprintf('Printing for same output\n');
PrintAverageKernelMatchings(AverageKernelMatchingsSameOut);
fprintf('Printing for all outpout\n');
PrintAverageKernelMatchings(AverageKernelMatchingsAllOut);

end

function MatchedAny = CheckCifar(ResultMat, GroundtTrurh)

MatchedAny = cellfun(@(x) str2double(x) == GroundtTrurh, ResultMat.predictions(:, 1));
% MatchedAny = strcmpi(char(GroundtTrurh), ResultMat.predictions(:, 1));

MatchedAny = MatchedAny';

end

function MatchedAny = CheckIlsvrc(ResultMat, GroundtTrurh)

% checking whether predictoin is correct
AcceptedResults = strsplit(GroundtTrurh, ', ');
prediction = ResultMat.predictions(:, 1);
MatchedAny = false(numel(prediction), 1);
for s = 1:numel(AcceptedResults)
  MatchedAny = strcmpi(AcceptedResults{s}, prediction) | MatchedAny;
end

MatchedAny = MatchedAny';

end

function PrintAverageKernelMatchings(AverageKernelMatchings, SpecificRange)

if nargin < 2
  SpecificRange = true;
end

[NumImages, NumContrasts] = size(AverageKernelMatchings.corrects);
NumLayers = size(AverageKernelMatchings.pixels.top.avg, 2);

%%
for WhichAnalysis = {'pixels', 'kernels'}
  AnalyseType = cell2mat(WhichAnalysis);
  fprintf('***Analyse: %s\n', AnalyseType);
  CurrentAnalysis = AverageKernelMatchings.(AnalyseType);
  SupportedFileds = fields(CurrentAnalysis);
  for at = 1:numel(SupportedFileds)
    AvgType = SupportedFileds{at};
    CurrentAnalysisType = CurrentAnalysis.(AvgType);
    fprintf('*******Type: %s\n', AvgType);
    for j = [0, 1, 2]
      switch j
        case 0
          fprintf('Network being incorrect\n');
          WhichResults = AverageKernelMatchings.corrects(:, NumContrasts) == j;
        case 1
          fprintf('Network being correct\n');
          WhichResults = AverageKernelMatchings.corrects(:, NumContrasts) == j;
        case 2
          WhichResults = true(NumImages, 1);
          fprintf('All results\n');
      end
      NonNaN = ~isnan(CurrentAnalysisType.avg(:, 1));
      WhichResults = WhichResults & NonNaN;
      if SpecificRange
        for i = 0:0.2:0.8
          if i ~= 0.80
            UpperBound = i + 0.2;
          else
            UpperBound = 1.0001;
          end
          RangeCondition = AverageKernelMatchings.scores(:, NumContrasts) >= i & AverageKernelMatchings.scores(:, NumContrasts) < UpperBound;
          meanvals = mean(CurrentAnalysisType.avg(WhichResults & RangeCondition, :), 1);
          fprintf(sprintf('>=%.2f<%.2f %s\n', i, UpperBound, repmat('%.2f ', [1, NumLayers])), meanvals);
        end
      else
        for i = [0:0.1:0.9, 0.999]
          meanvals = mean(CurrentAnalysisType.avg(AverageKernelMatchings.scores(:, NumContrasts) >= i & WhichResults, :), 1);
          fprintf(sprintf('>=%.2f %s\n', i, repmat('%.2f ', [1, NumLayers])), meanvals);
        end
      end
    end
  end
end

end
