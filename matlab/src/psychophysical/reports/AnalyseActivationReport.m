function [AverageKernelMatchingsEqTop, AverageKernelMatchingsAll] = AnalyseActivationReport(ActivationReportPath)
%AnalyseActivationReport Summary of this function goes here
%   Detailed explanation goes here

% for now we only have groundtruth for this dataset.
DatasetName = 'ilsvrc-test';
if strcmpi(DatasetName, 'ilsvrc-test')
  ValidationDir = '/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2012/validation/';
  
  labels = dlmread(sprintf('%sILSVRC2012_validation_ground_truth.txt', ValidationDir));
  ImageInfos = load(sprintf('%sILSVRC2012_validation_meta.mat', ValidationDir));
else
  error('No groundtruth is available for dataset: %s\n', DatasetName);
end

outdir = fileparts(ActivationReportPath);

%% Creating the matrix contrast versus accuracy

AverageKernelMatchingsEqTopPath = [outdir, '/AverageKernelMatchingsEqTop.mat'];
AverageKernelMatchingsAllPath = [outdir, '/AverageKernelMatchingsAll.mat'];
if exist(AverageKernelMatchingsEqTopPath, 'file')
  ActivationReport = load(ActivationReportPath);
  ActivationReport = ActivationReport.ActivationReport;
  
  NumImages = numel(ActivationReport);
  
  [nContrasts, ~, NumLayers] = size(ActivationReport(1).CompMatrix);
  
  EqTopAvgs = zeros(NumImages, NumLayers);
  AllAvgs = zeros(NumImages, NumLayers);
  predictions = cell(NumImages, 1);
  corrects = zeros(NumImages, nContrasts);
  scores = zeros(NumImages, nContrasts);
  parfor i = 1:NumImages
    EqTopTmp = ContrastVsAccuracy(ActivationReport(i), false);
    EqTopAvgs(i, :) = EqTopTmp.avg;
    predictions{i} = EqTopTmp.predictions;
    
    AllTmp = ContrastVsAccuracy(ActivationReport(i), true);
    AllAvgs(i, :) = AllTmp.avg;
    
    % checking whether predictoin is correct
    AcceptedResults = strsplit(ImageInfos.synsets(labels(i)).words, ', ');
    prediction = EqTopTmp.predictions(:, 1);
    MatchedAny = false(nContrasts, 1);
    for s = 1:numel(AcceptedResults)
      MatchedAny = strcmpi(AcceptedResults{s}, prediction) | MatchedAny;
    end
    
    corrects(i, :) = MatchedAny';
    scores(i, :) = cell2mat(EqTopTmp.predictions(:, 2))';
  end
  
  % for those equal to top contrast
  AverageKernelMatchingsEqTop.avgs = EqTopAvgs;
  AverageKernelMatchingsEqTop.predictions = predictions;
  AverageKernelMatchingsEqTop.corrects = corrects;
  AverageKernelMatchingsEqTop.scores = scores;
  
  % for all
  AverageKernelMatchingsAll.avgs = AllAvgs;
  AverageKernelMatchingsAll.predictions = predictions;
  AverageKernelMatchingsAll.corrects = corrects;
  AverageKernelMatchingsAll.scores = scores;
  
  save(AverageKernelMatchingsEqTopPath, 'AverageKernelMatchingsEqTop');
  save(AverageKernelMatchingsAllPath, 'AverageKernelMatchingsAll');
else
  AverageKernelMatchingsEqTop = load(AverageKernelMatchingsEqTopPath);
  AverageKernelMatchingsEqTop = AverageKernelMatchingsEqTop.AverageKernelMatchingsEqTop;
  
  AverageKernelMatchingsAll = load(AverageKernelMatchingsAllPath);
  AverageKernelMatchingsAll = AverageKernelMatchingsAll.AverageKernelMatchingsAll;
end

%% printing the result according to being correct or not
% fprintf('Printing for all\n');
% PrintAverageKernelMatchings(AverageKernelMatchingsAll);
fprintf('Printing for top\n');
PrintAverageKernelMatchings(AverageKernelMatchingsEqTop);

end

function PrintAverageKernelMatchings(AverageKernelMatchings, SpecificRange)

if nargin < 2
  SpecificRange = true;
end

[NumImages, NumContrasts] = size(AverageKernelMatchings.corrects);
NumLayers = size(AverageKernelMatchings.avgs, 2);

%%
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
  if SpecificRange
    for i = 0:0.2:0.8
      if i ~= 0.80
        UpperBound = i + 0.2;
      else
        UpperBound = 1.0001;
      end
      RangeCondition = AverageKernelMatchings.scores(:, NumContrasts) >= i & AverageKernelMatchings.scores(:, NumContrasts) < UpperBound;
      meanvals = mean(AverageKernelMatchings.avgs(WhichResults & RangeCondition, :));
      fprintf(sprintf('>=%.2f<%.2f %s\n', i, UpperBound, repmat('%.2f ', [1, NumLayers])), meanvals);
    end
  else
    for i = [0:0.1:0.9, 0.999]
      meanvals = mean(AverageKernelMatchings.avgs(AverageKernelMatchings.scores(:, NumContrasts) >= i & WhichResults, :));
      fprintf(sprintf('>=%.2f %s\n', i, repmat('%.2f ', [1, NumLayers])), meanvals);
    end
  end
end

end
