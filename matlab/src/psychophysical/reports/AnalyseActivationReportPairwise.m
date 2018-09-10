function PairwiseReport = AnalyseActivationReportPairwise(ActivationReportPath, DatasetName)
%AnalyseActivationReportPairwise Summary of this function goes here
%   Detailed explanation goes here

outdir = fileparts(ActivationReportPath);

%% Creating the matrix contrast versus accuracy

PairwiseReportpPath = [outdir, '/PairwiseReport.mat'];
if ~exist(PairwiseReportpPath, 'file')
  ActivationReport = load(ActivationReportPath);
  ActivationReport = ActivationReport.ActivationReport;
  
  PairwiseReport.same = ProcessOneType(ActivationReport, 'same');
  PairwiseReport.diff = ProcessOneType(ActivationReport, 'diff');
  PairwiseReport.all = ProcessOneType(ActivationReport, 'all');
  
  save(PairwiseReportpPath, 'PairwiseReport');
else
  PairwiseReport = load(PairwiseReportpPath);
  PairwiseReport = PairwiseReport.PairwiseReport;
end

%% printing the results
AverageKernelMatchings = AnalyseActivationReport(ActivationReportPath, DatasetName, false);
corrects = AverageKernelMatchings.same.out.corrects(:, end);

fprintf('**** Same results\n');
PrintOneType(PairwiseReport.same, corrects);
fprintf('**** Different results\n');
PrintOneType(PairwiseReport.diff, corrects);
fprintf('**** All results\n');
PrintOneType(PairwiseReport.all, corrects);

end

function PrintOneType(PairwiseReport, corrects)

metrices = fields(PairwiseReport.metrices);
nMetrices = numel(metrices);
for i = 1:nMetrices
  fprintf('**** Results for %s ****\n', metrices{i});
  fprintf('- All\n');
  PrintAverageKernelMatchings(PairwiseReport.metrices.(metrices{i}).avg);
  fprintf('-Corrects\n');
  PrintAverageKernelMatchings(PairwiseReport.metrices.(metrices{i}).avg, corrects);
end

end

function PairwiseReport = ProcessOneType(ActivationReport, WhichRype)

data = ActivationReport.data;
nImages = ActivationReport.info.nImages;
nContrasts = ActivationReport.info.nContrasts;
nLayers = ActivationReport.info.nLayers;

nComparisons = nContrasts - 1;

metrices = fields(data{1}.metrices);
nMetrices = numel(metrices);
for i = 1:nMetrices
  PairwiseReport.metrices.(metrices{i}).avg = zeros(nImages, nComparisons, nLayers);
end

for i = 1:nImages
  for t = 1:nComparisons
    ComparisonReport = ContrastVsAccuracy(data{i}, WhichRype, [1:t - 1, t + 1:nComparisons]);
    
    for k = 1:nMetrices
      PairwiseReport.metrices.(metrices{k}).avg(i, t, :) = ComparisonReport.metrices.(metrices{k}).avg;
    end
  end
end

end

function PrintAverageKernelMatchings(PairwiseReport, corrects)

[nImages, nComparisons, nLayers] = size(PairwiseReport);

if nargin < 2
  corrects = true(nImages, 1);
end

for i = 1:nComparisons
  NonNaN = ~isnan(PairwiseReport(:, i, 1));
  meanvals = mean(PairwiseReport(NonNaN & corrects, i, :), 1);
  stdvals = std(PairwiseReport(NonNaN & corrects, i, :), [], 1);
  PrintVals(1:2:nLayers * 2) = permute(meanvals, [3, 1, 2]);
  PrintVals(2:2:nLayers * 2) = permute(stdvals, [3, 1, 2]);
  fprintf(sprintf('%d | %s\n', sum(NonNaN & corrects), repmat('%.2f(%.2f) ', [1, nLayers])), PrintVals);
end

end
