function AverageKernelMatching = ContrastVsAccuracy(ActivationReport, ConsiderAll, ExcludeList)
%ContrastVsAccuracy Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
  ExcludeList = [];
end
if nargin < 2
  ConsiderAll = true;
end

ContrastNames = fieldnames(ActivationReport.cls);
nContrasts = numel(ContrastNames);

predictions = cell(nContrasts, 2);

for i = 1:nContrasts
  predictions{i, 1} = ActivationReport.cls.(ContrastNames{i}).prediction.type;
  predictions{i, 2} = ActivationReport.cls.(ContrastNames{i}).prediction.score;
end

if ConsiderAll
  IndsCorrectlyClassified = true(nContrasts, 1);
else
  ClassType = predictions{nContrasts, 1};
  
  IndsCorrectlyClassified = strcmp(predictions(:, 1), ClassType);
end
IndsCorrectlyClassified(ExcludeList) = false;

CompPixelsTop = ActivationReport.CompMatrix(IndsCorrectlyClassified, IndsCorrectlyClassified, :);
CompPixelsHist = ActivationReport.CompMatrixHist(IndsCorrectlyClassified, IndsCorrectlyClassified, :);
CompMatrixHistKernels = ActivationReport.kernels.CompMatrixHist(IndsCorrectlyClassified, IndsCorrectlyClassified, :);
[nSelected, ~, nLayers] = size(CompPixelsTop);

nElements = nSelected * (nSelected - 1) / 2;
if nElements == 0
  PixelsTop = SetToNaN(nLayers);
  PixelsHist = PixelsTop;
  KernelsHist = PixelsTop;
else
  NonZeroColumns = [];
  
  [PixelsTop, NonZeroColumns] = StatisticsHelper(CompPixelsTop, NonZeroColumns);
  [PixelsHist, NonZeroColumns] = StatisticsHelper(CompPixelsHist, NonZeroColumns);
  [KernelsHist, ~] = StatisticsHelper(CompMatrixHistKernels, NonZeroColumns);
end

AverageKernelMatching.pixels.top = PixelsTop;
AverageKernelMatching.pixels.hist = PixelsHist;
AverageKernelMatching.kernels.hist = KernelsHist;
AverageKernelMatching.predictions = predictions;

end

function ReportStatistics = SetToNaN(nLayers)

ReportStatistics.avg = nan(1, nLayers);
ReportStatistics.std = nan(1, nLayers);
ReportStatistics.max = nan(1, nLayers);
ReportStatistics.min = nan(1, nLayers);

end

function [ReportStatistics, NonZeroColumns] = StatisticsHelper(ReportMat, NonZeroColumns)

[rows, cols, chns] = size(ReportMat);
ReportMat = reshape(ReportMat, rows * cols, chns);
if isempty(NonZeroColumns)
  NonZeroColumns = sum(ReportMat, 2);
  NonZeroColumns = NonZeroColumns > 0;
end
ReportMat = ReportMat(NonZeroColumns, :);

ReportStatistics.avg = mean(ReportMat, 1);
ReportStatistics.std = std(ReportMat, [], 1);
ReportStatistics.max = max(ReportMat, [], 1);
ReportStatistics.min = min(ReportMat, [], 1);

end
