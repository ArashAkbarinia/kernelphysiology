function PairwiseReport = AnalyseActivationReportPairwise(ActivationReportPath, DatasetName, MinConfidence)
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
  PairwiseReport.similar = ProcessOneType(ActivationReport, 'similar');
  PairwiseReport.dissimilar = ProcessOneType(ActivationReport, 'dissimilar');
else
  PairwiseReport = load(PairwiseReportpPath);
  PairwiseReport = PairwiseReport.PairwiseReport;
end

%% printing the results
AverageKernelMatchings = AnalyseActivationReport(ActivationReportPath, DatasetName, false);
corrects = AverageKernelMatchings.same.out.corrects;
if nargin < 3
  MinConfidence = 0.5;
end
scores = AverageKernelMatchings.same.out.scores > MinConfidence;

fprintf('**** Same results\n');
PairwiseReport.same.reports = PrintOneType(PairwiseReport.same, corrects, scores);
fprintf('**** Different results\n');
PairwiseReport.diff.reports = PrintOneType(PairwiseReport.diff, corrects, scores);
fprintf('**** All results\n');
PairwiseReport.all.reports = PrintOneType(PairwiseReport.all, corrects, scores);
fprintf('**** Similar results\n');
PairwiseReport.similar.reports = PrintOneType(PairwiseReport.similar, corrects, scores);
fprintf('**** Dissimilar results\n');
PairwiseReport.dissimilar.reports = PrintOneType(PairwiseReport.dissimilar, corrects, scores);

if ~exist(PairwiseReportpPath, 'file')
  save(PairwiseReportpPath, 'PairwiseReport');
end

end

function TableReports = PrintOneType(PairwiseReport, corrects, scores)

metrices = fields(PairwiseReport.metrices);
nMetrices = numel(metrices);
for i = 1:nMetrices
  fprintf('**** Results for %s ****\n', metrices{i});
  fprintf('- All\n');
  TableReports.(metrices{i}).all = PrintAverageKernelMatchings(PairwiseReport.metrices.(metrices{i}).avg);
  fprintf('- Corrects\n');
  TableReports.(metrices{i}).corrects = PrintAverageKernelMatchings(PairwiseReport.metrices.(metrices{i}).avg, corrects);
  fprintf('- Scores\n');
  TableReports.(metrices{i}).scores = PrintAverageKernelMatchings(PairwiseReport.metrices.(metrices{i}).avg, scores);
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

function TableReports = PrintAverageKernelMatchings(PairwiseReport, WhichImages)

[nImages, nComparisons, nLayers] = size(PairwiseReport);

% TODO: if it becomes relevant consider other results as well.
TopXPred = 1;
if nargin < 2
  WhichImages = true(nImages, nComparisons);
end

TableReports.avg = zeros(nComparisons, nLayers);
TableReports.std = zeros(nComparisons, nLayers);
TableReports.nSamples = zeros(nComparisons, 1);

for i = 1:nComparisons
  NonNaN = ~isnan(PairwiseReport(:, i, 1));
  TmpWhich = WhichImages(:, i, TopXPred);
  meanvals = mean(PairwiseReport(NonNaN & TmpWhich, i, :), 1);
  TableReports.avg(i, :) = meanvals;
  stdvals = std(PairwiseReport(NonNaN & TmpWhich, i, :), [], 1);
  TableReports.std(i, :) = stdvals;
  PrintVals(1:2:nLayers * 2) = permute(meanvals, [3, 1, 2]);
  PrintVals(2:2:nLayers * 2) = permute(stdvals, [3, 1, 2]);
  TableReports.nSamples(i, 1) = sum(NonNaN & TmpWhich);
  fprintf(sprintf('%d | %s\n', sum(NonNaN & TmpWhich), repmat('%.2f(%.2f) ', [1, nLayers])), PrintVals);
end

end
