function AverageKernelMatching = ContrastVsAccuracy(ActivationReport, ConsiderAll)
%ContrastVsAccuracy Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
  ConsiderAll = true;
end

ContrastNames = fieldnames(ActivationReport.cls);
nContrasts = numel(ContrastNames);

predictioins = cell(nContrasts, 2);

for i = 1:nContrasts
  predictioins{i, 1} = ActivationReport.cls.(ContrastNames{i}).prediction.type;
  predictioins{i, 2} = ActivationReport.cls.(ContrastNames{i}).prediction.score;
end

if ConsiderAll
  IndsCorrectlyClassified = true(nContrasts, 1);
else
  ClassType = predictioins{nContrasts, 1};
  
  IndsCorrectlyClassified = strcmp(predictioins(:, 1), ClassType);
end

CompMatrix = ActivationReport.CompMatrix(IndsCorrectlyClassified, IndsCorrectlyClassified, :);
nSelected = size(CompMatrix, 1);

nElements = nSelected * (nSelected - 1) / 2;
if nElements == 0
  AverageKernelMatching = zeros(1, 5);
else
  AverageKernelMatching = sum(sum(CompMatrix)) ./ nElements;
  AverageKernelMatching = permute(AverageKernelMatching, [1, 3, 2]);
end

AverageKernelMatching(end + 1) = predictioins{10, 2};

end
