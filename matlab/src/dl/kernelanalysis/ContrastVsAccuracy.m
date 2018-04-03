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
AverageKernelMatching = permute(AverageKernelMatching, [1, 3, 2]);

StrFormat = repmat('%.2f ', [1, size(AverageKernelMatching, 2)]);
StrFormat = [StrFormat, '%.2f\n'];
fprintf(StrFormat, AverageKernelMatching, predictioins{10, 2});

AverageKernelMatching(end + 1) = predictioins{10, 2};

end
