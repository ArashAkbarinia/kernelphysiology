function AverageKernelMatching = ContrastVsAccuracy(ActivationReport)
%ContrastVsAccuracy Summary of this function goes here
%   Detailed explanation goes here

ContrastNames = fieldnames(ActivationReport.cls);
nContrasts = numel(ContrastNames);

predictioins = cell(nContrasts, 2);

for i = 1:nContrasts
  predictioins{i, 1} = ActivationReport.cls.(ContrastNames{i}).prediction.type;
  predictioins{i, 2} = ActivationReport.cls.(ContrastNames{i}).prediction.score;
end

ClassType = predictioins{nContrasts, 1};

IndsCorrectlyClassified = strcmp(predictioins(:, 1), ClassType);

CompMatrix = ActivationReport.CompMatrix(IndsCorrectlyClassified, IndsCorrectlyClassified, :);
nSelected = size(CompMatrix, 1);

nElements = nSelected * (nSelected - 1) / 2;
AverageKernelMatching = sum(sum(CompMatrix)) ./ nElements;

StrFormat = repmat('%.2f ', [1, size(AverageKernelMatching, 3)]);
StrFormat = [StrFormat, '\n'];
fprintf(StrFormat, AverageKernelMatching);

end
