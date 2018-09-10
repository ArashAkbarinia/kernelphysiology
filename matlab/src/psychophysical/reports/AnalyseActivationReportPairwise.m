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
[AverageKernelMatchings, ~] = AnalyseActivationReport(ActivationReportPath, DatasetName, false);
corrects = AverageKernelMatchings.corrects(:, end);

fprintf('**** Same results\n');
PrintOneType(PairwiseReport.same, corrects);
fprintf('**** Different results\n');
PrintOneType(PairwiseReport.diff, corrects);
fprintf('**** All results\n');
PrintOneType(PairwiseReport.all, corrects);

end

function PrintOneType(PairwiseReport, corrects)

fprintf('Printing for top pixels\n');
fprintf('- All\n');
PrintAverageKernelMatchings(PairwiseReport.pixels.top.avg);
fprintf('-Corrects\n');
PrintAverageKernelMatchings(PairwiseReport.pixels.top.avg, corrects);
fprintf('Printing for hist pixels\n');
fprintf('All\n');
PrintAverageKernelMatchings(PairwiseReport.pixels.hist.avg);
fprintf('-Corrects\n');
PrintAverageKernelMatchings(PairwiseReport.pixels.hist.avg, corrects);
fprintf('Printing for hist kernels\n');
fprintf('All\n');
PrintAverageKernelMatchings(PairwiseReport.kernels.hist.avg);
fprintf('-Corrects\n');
PrintAverageKernelMatchings(PairwiseReport.kernels.hist.avg, corrects);

end

function PairwiseReport = ProcessOneType(ActivationReport, WhichRype)

NumImages = numel(ActivationReport);

[nContrasts, ~, NumLayers] = size(ActivationReport{1}.CompMatrix);

nComparisons = nContrasts - 1;
SameOutPixelsTop = zeros(NumImages, nComparisons, NumLayers);
SameOutPixelsHist = zeros(NumImages, nComparisons, NumLayers);
SameOutKernelsHist = zeros(NumImages, nComparisons, NumLayers);

parfor i = 1:NumImages
  for t = 1:nComparisons
    SameOutCompare = ContrastVsAccuracy(ActivationReport{i}, WhichRype, [1:t - 1, t + 1:nComparisons]);
    
    SameOutPixelsTop(i, t, :) = SameOutCompare.pixels.top.avg;
    SameOutPixelsHist(i, t, :) = SameOutCompare.pixels.hist.avg;
    SameOutKernelsHist(i, t, :) = SameOutCompare.kernels.hist.avg;
  end
end

PairwiseReport.pixels.top.avg = SameOutPixelsTop;
PairwiseReport.pixels.hist.avg = SameOutPixelsHist;
PairwiseReport.kernels.hist.avg = SameOutKernelsHist;

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
