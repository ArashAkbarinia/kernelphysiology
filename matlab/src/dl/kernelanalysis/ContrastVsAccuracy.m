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

CompMatrix = ActivationReport.CompMatrix(IndsCorrectlyClassified, IndsCorrectlyClassified, :);
[nSelected, ~, nLayers] = size(CompMatrix);

nElements = nSelected * (nSelected - 1) / 2;
if nElements == 0
  MeanVal = zeros(1, nLayers);
else
  MeanVal = sum(sum(CompMatrix)) ./ nElements;
  MeanVal = permute(MeanVal, [1, 3, 2]);
end

AverageKernelMatching.avg = MeanVal;
AverageKernelMatching.predictions = predictions;

end
