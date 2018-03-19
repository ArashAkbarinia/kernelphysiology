function [ ] = CnnEvalImage( net, im )
%CnnEvalImage  evaluates the input image with given CNN network.
%   Detailed explanation goes here

% stdim = rgb2gray(im);

stdim = StandardImage(im);

im = gpuArray(im);
res = vl_simplenn(net, im);

scores = squeeze(gather(res(end).x));
[~, best] = max(scores);

EvaluationReport(i) = best == LabelSet(i);

end
