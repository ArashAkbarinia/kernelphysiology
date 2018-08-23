function EvaluationReport = CnnCifarEval(net, imdb, ContrastLevels)
%CnnCifarEval Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
  ContrastLevels = [1, 3, 5, 7, 10, 13, 15, 30, 50, 75, 100];
end

images = imdb.images.data;
labels = imdb.images.labels;
imsets = imdb.images.set;

ValSetInds = imsets == 3;

LabelSet = labels(ValSetInds)';
TestSet = images(:, :, :, ValSetInds);

nContrasts = numel(ContrastLevels);
AllAccuracies = zeros(1, nContrasts);
for c = 1:nContrasts
  contrast = ContrastLevels(c);
  % converting it to uint8 because the imdb is stored in single
  ContrastedImages = AdjustContrast(uint8(TestSet), contrast / 100);
  % dont't convert it back to uint8 because keras has been trained [0, 1]
  % ContrastedImages = uint8(ContrastedImages .* 255);
  
  NetPredictions = net.classify(ContrastedImages);
  PredictionEval = int8(NetPredictions) == LabelSet;
  
  ContrastName = sprintf('c%.3u', contrast);
  
  EvaluationReport.cls.(ContrastName).NetPredictions = NetPredictions;
  EvaluationReport.cls.(ContrastName).PredictionEval = PredictionEval;
  AllAccuracies(1, c) = mean(PredictionEval);
end

EvaluationReport.sum.AllAccuracies = AllAccuracies;

end
