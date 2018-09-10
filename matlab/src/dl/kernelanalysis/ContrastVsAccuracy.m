function ComparisonReport = ContrastVsAccuracy(ActivationReport, WhichResults, ExcludeList)
%ContrastVsAccuracy Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
  ExcludeList = [];
end
if nargin < 2
  WhichResults = 'all';
end

ContrastNames = fieldnames(ActivationReport.cls);
nContrasts = numel(ContrastNames);

predictions = cell(nContrasts, 2);

for i = 1:nContrasts
  predictions{i, 1} = ActivationReport.cls.(ContrastNames{i}).prediction.type;
  predictions{i, 2} = ActivationReport.cls.(ContrastNames{i}).prediction.score;
end

if strcmpi(WhichResults, 'all')
  IndsCorrectlyClassified = true(nContrasts, 1);
elseif strcmpi(WhichResults, 'same')
  ClassType = predictions{nContrasts, 1};
  IndsCorrectlyClassified = strcmpi(predictions(:, 1), ClassType);
elseif strcmpi(WhichResults, 'diff')
  ClassType = predictions{nContrasts, 1};
  IndsCorrectlyClassified = ~strcmpi(predictions(:, 1), ClassType);
  IndsCorrectlyClassified(nContrasts) = true;
end
IndsCorrectlyClassified(ExcludeList) = false;

metrices = fields(ActivationReport.metrices);

NonZeroColumns = [];
for i = 1:numel(metrices)
  CurrentMetricName = metrices{i};
  CurrentMetricMat = ActivationReport.metrices.(CurrentMetricName)(IndsCorrectlyClassified, IndsCorrectlyClassified, :);
  
  [nSelected, ~, nLayers] = size(CurrentMetricMat);
  
  nElements = nSelected * (nSelected - 1) / 2;
  if nElements == 0
    CurrentMetricReport = SetToNaN(nLayers);
  else
    [CurrentMetricReport, NonZeroColumns] = StatisticsHelper(CurrentMetricMat, NonZeroColumns);
  end
  
  ComparisonReport.metrices.(CurrentMetricName) = CurrentMetricReport;
end

ComparisonReport.predictions = predictions;

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
