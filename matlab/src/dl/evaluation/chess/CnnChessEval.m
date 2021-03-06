function EvaluationReport = CnnChessEval( net, imdb )
%CNNCIFAREVAL Summary of this function goes here
%   Detailed explanation goes here

images = imdb.images.data;
labels = imdb.images.labels;
imsets = imdb.images.set;

net = vl_simplenn_tidy(net);
net = vl_simplenn_move(net, 'gpu');

ValSetInds = imsets == 3;
TestSet = images(:, :, :, ValSetInds);
LabelSet = labels(ValSetInds);

nimages = size(TestSet, 4);

EvaluationReport = zeros(nimages, 5);
% running the CNN
for i = 1:nimages
  im = TestSet(:, :, :, i);
  im = gpuArray(im);
  res = vl_simplenn(net, im);
  
  scores = squeeze(gather(res(end).x));
  [~, best] = max(scores);
  
  EvaluationReport(i, 1) = best == LabelSet(i);
  EvaluationReport(i, 2) = best;
  EvaluationReport(i, 3:5) = scores';
end

end
