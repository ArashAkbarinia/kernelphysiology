function AllContrastActivationReport = AnalyseActivationReport(ActivationReportPath, DatasetName, PrintResults)
%AnalyseActivationReport Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
  PrintResults = true;
end

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

AllContrastActivationReportPath = [outdir, '/AllContrastActivationReport.mat'];
if ~exist(AllContrastActivationReportPath, 'file')
  ActivationReport = load(ActivationReportPath);
  ActivationReport = ActivationReport.ActivationReport;
  
  NumImages = numel(ActivationReport.data);
  
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
    
  CheckGT.CompareToGT = true;
  CheckGT.GroundTruths = GroundTruths;
  CheckGT.imdb = imdb;
  AllContrastActivationReport.same = ProcessOneType(ActivationReport, 'same', CheckGT);
  CheckGT.CompareToGT = false;
  AllContrastActivationReport.diff = ProcessOneType(ActivationReport, 'diff', CheckGT);
  AllContrastActivationReport.all = ProcessOneType(ActivationReport, 'all', CheckGT);
  
  AllContrastActivationReport.diff.out = AllContrastActivationReport.same.out;
  AllContrastActivationReport.all.out = AllContrastActivationReport.same.out;
  
  save(AllContrastActivationReportPath, 'AllContrastActivationReport');
else
  AllContrastActivationReport = load(AllContrastActivationReportPath);
  AllContrastActivationReport = AllContrastActivationReport.AllContrastActivationReport;
end

%% printing the result according to being correct or not
if PrintResults
  fprintf('**** Same results\n');
  PrintAverageKernelMatchings(AllContrastActivationReport.same);
  fprintf('**** Different results\n');
  PrintAverageKernelMatchings(AllContrastActivationReport.diff);
  fprintf('**** All results\n');
  PrintAverageKernelMatchings(AllContrastActivationReport.all);
end

end

function AllContrastReport = ProcessOneType(ActivationReport, WhichRype, CheckGT)

data = ActivationReport.data;
nImages = ActivationReport.info.nImages;
nContrasts = ActivationReport.info.nContrasts;
nLayers = ActivationReport.info.nLayers;

metrices = fields(data{1}.metrices);
nMetrices = numel(metrices);
for i = 1:nMetrices
  AllContrastReport.metrices.(metrices{i}).avg = zeros(nImages, nLayers);
end

AllContrastReport.out.predictions = cell(nImages, 1);
AllContrastReport.out.corrects = zeros(nImages, nContrasts);
AllContrastReport.out.scores = zeros(nImages, nContrasts);

for i = 1:nImages
  ComparisonReport = ContrastVsAccuracy(data{i}, WhichRype);
  AllContrastReport.out.predictions{i, 1} = ComparisonReport.predictions;
  
  for k = 1:nMetrices
    AllContrastReport.metrices.(metrices{k}).avgs(i, :) = ComparisonReport.metrices.(metrices{k}).avg;
  end
  
  if CheckGT.CompareToGT
    if ~isempty(CheckGT.imdb)
      MatchedAny = CheckCifar(ComparisonReport.predictions, CheckGT.GroundTruths(i));
    else
      MatchedAny = CheckIlsvrc(ComparisonReport.predictions, CheckGT.GroundTruths{i});
    end
    AllContrastReport.out.corrects(i, :) = MatchedAny';
    AllContrastReport.out.scores(i, :) = cell2mat(ComparisonReport.predictions(:, 2))';
  end
end

end

function MatchedAny = CheckCifar(predictions, GroundtTrurh)

MatchedAny = cellfun(@(x) str2double(x) == GroundtTrurh, predictions(:, 1));
% MatchedAny = strcmpi(char(GroundtTrurh), ResultMat.predictions(:, 1));

MatchedAny = MatchedAny';

end

function MatchedAny = CheckIlsvrc(NetPredictions, GroundtTrurh)

% checking whether predictoin is correct
AcceptedResults = strsplit(GroundtTrurh, ', ');
prediction = NetPredictions(:, 1);
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

NetPred = AverageKernelMatchings.out;

%%
metrices = AverageKernelMatchings.metrices;
MetricNames = fields(metrices);

[NumImages, NumContrasts] = size(NetPred.corrects);
NumLayers = size(metrices.(MetricNames{1}).avg, 2);

for at = 1:numel(MetricNames)
  AvgType = MetricNames{at};
  CurrentAnalysisType = metrices.(AvgType);
  fprintf('*******Type: %s\n', AvgType);
  for j = [0, 1, 2]
    switch j
      case 0
        fprintf('Network being incorrect\n');
        WhichResults = NetPred.corrects(:, NumContrasts) == j;
      case 1
        fprintf('Network being correct\n');
        WhichResults = NetPred.corrects(:, NumContrasts) == j;
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
        RangeCondition = NetPred.scores(:, NumContrasts) >= i & NetPred.scores(:, NumContrasts) < UpperBound;
        meanvals = mean(CurrentAnalysisType.avg(WhichResults & RangeCondition, :), 1);
        fprintf(sprintf('>=%.2f<%.2f %s\n', i, UpperBound, repmat('%.2f ', [1, NumLayers])), meanvals);
      end
    else
      for i = [0:0.1:0.9, 0.999]
        meanvals = mean(CurrentAnalysisType.avg(NetPred.scores(:, NumContrasts) >= i & WhichResults, :), 1);
        fprintf(sprintf('>=%.2f %s\n', i, repmat('%.2f ', [1, NumLayers])), meanvals);
      end
    end
  end
end

end
