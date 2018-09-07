function PairwiseReport = AnalyseActivationReportPairwise(ActivationReportPath)
%AnalyseActivationReportPairwise Summary of this function goes here
%   Detailed explanation goes here

outdir = fileparts(ActivationReportPath);

%% Creating the matrix contrast versus accuracy

PairwiseReportpPath = [outdir, '/PairwiseReport.mat'];
if ~exist(PairwiseReportpPath, 'file')
  ActivationReport = load(ActivationReportPath);
  ActivationReport = ActivationReport.ActivationReport;
  
  NumImages = numel(ActivationReport);
  
  [nContrasts, ~, NumLayers] = size(ActivationReport{1}.CompMatrix);
  
  nComparisons = nContrasts - 1;
  SameOutPixelsTop = zeros(NumImages, nComparisons, NumLayers);
  SameOutPixelsHist = zeros(NumImages, nComparisons, NumLayers);
  SameOutKernelsHist = zeros(NumImages, nComparisons, NumLayers);
  
  for i = 1:NumImages
    for t = 1:nComparisons
      SameOutCompare = ContrastVsAccuracy(ActivationReport{i}, false, [1:t - 1, t + 1:nComparisons]);
      
      SameOutPixelsTop(i, t, :) = SameOutCompare.pixels.top.avg;
      SameOutPixelsHist(i, t, :) = SameOutCompare.pixels.hist.avg;
      SameOutKernelsHist(i, t, :) = SameOutCompare.kernels.hist.avg;
    end
  end
  
  PairwiseReport.pixels.top.avg = SameOutPixelsTop;
  PairwiseReport.pixels.hist.avg = SameOutPixelsHist;
  PairwiseReport.kernels.hist.avg = SameOutKernelsHist;
  
  save(PairwiseReportpPath, 'PairwiseReport');
else
  PairwiseReport = load(PairwiseReportpPath);
  PairwiseReport = PairwiseReport.PairwiseReport;
end

%% printing the results
fprintf('Printing for top pixels\n');
PrintAverageKernelMatchings(PairwiseReport.pixels.top.avg);
fprintf('Printing for hist pixels\n');
PrintAverageKernelMatchings(PairwiseReport.pixels.hist.avg);
fprintf('Printing for hist kernels\n');
PrintAverageKernelMatchings(PairwiseReport.kernels.hist.avg);

end

function PrintAverageKernelMatchings(PairwiseReport)

[~, nComparisons, nLayers] = size(PairwiseReport);

for i = 1:nComparisons
  NonNaN = ~isnan(PairwiseReport(:, i, 1));
  meanvals = mean(PairwiseReport(NonNaN, i, :), 1);
  fprintf(sprintf('%s\n', repmat('%.2f ', [1, nLayers])), meanvals);
end

end
