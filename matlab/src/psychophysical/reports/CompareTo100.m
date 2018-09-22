function CompareTo100Report = CompareTo100(ActivationReport)
%COMPARETO100 Summary of this function goes here
%   Detailed explanation goes here

ContrastNames = fieldnames(ActivationReport.cls);
nContrasts = numel(ContrastNames);

nTopXPred = 1;%numel(ActivationReport.cls.(ContrastNames{1}).prediction.type);
types = cell(nContrasts, nTopXPred);
scores = zeros(nContrasts, nTopXPred);

for i = 1:nContrasts
  types{i, :} = ActivationReport.cls.(ContrastNames{i}).prediction.type';
%   scores(i, :) = ActivationReport.cls.(ContrastNames{i}).prediction.score';
end

WhichResults = 'same';

if strcmpi(WhichResults, 'same')
  ClassType = types{nContrasts, 1}';
  CompareTo100Report = strcmpi(types(:, 1), ClassType);
elseif strcmpi(WhichResults, 'similar')
  CompareTo100Report = GetSimilarTypesInds(types, nContrasts);
elseif strcmpi(WhichResults, 'dissimilar')
  CompareTo100Report = ~GetSimilarTypesInds(types, nContrasts);
  CompareTo100Report(nContrasts) = true;
elseif strcmpi(WhichResults, 'diff')
  ClassType = types{nContrasts, 1};
  CompareTo100Report = ~strcmpi(types(:, 1), ClassType);
  CompareTo100Report(nContrasts) = true;
end

end

function IndsCorrectlyClassified = GetSimilarTypesInds(types, nContrasts)

ClassType = types(nContrasts, :);
IndsCorrectlyClassified = true(nContrasts, 1);
for i = 1:nContrasts - 1
  IndsCorrectlyClassified(i, 1) = any(strcmpi(types(i, 1), ClassType));
end

end
