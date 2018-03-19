function FigureHandler = PlotTopCorrelatedKernels( CorrMat, net, topn, isvisible )
%PlotTopCorrelatedKernels Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
  topn = 25;
end
if nargin < 4
  isvisible = 'on';
end

FigRows = round(sqrt(topn));
FigCols = ceil(sqrt(topn));

[kers, ~, chns] = size(CorrMat);

LowerMat = logical(tril(ones(kers, kers)));

FigureHandler = zeros(chns);
for c = 1:chns% change to 3 later on
  FigureHandler(c) = figure('name', ['Similar kernels channel ', num2str(c)], 'visible', isvisible);
  CorrChn = CorrMat(:, :, c);
  CorrChn(LowerMat) = -inf;
  [SortedCorrs, SortedInds] = sort(CorrChn(:), 'descend');
  [rows, cols]= ind2sub([kers, kers], SortedInds(1:topn));
  splotind = 1;
  for i = 1:topn
    subplot(FigRows, FigCols, splotind);
    splotind = splotind + 1;
%     ker1 = net.layers{1, 1}.weights{1, 1}(:, :, c, rows(i));
%     ker2 = net.layers{1, 1}.weights{1, 1}(:, :, c, cols(i));
    ker1 = net(:, :, :, rows(i));
    ker2 = net(:, :, :, cols(i));
    
%     ker1 = NormaliseChannel(ker1);
%     ker2 = NormaliseChannel(ker2);
%     ker1(ker1 < 0) = -1;
%     ker1(ker1 > 0) = +1;
%     ker2(ker2 < 0) = -1;
%     ker2(ker2 > 0) = +1;
    
    imshowpair(ker1, ker2, 'montage');
    title(['Kernels [', num2str(rows(i)), ', ', num2str(cols(i)), '] Corr = ', num2str(SortedCorrs(i), 2)]);
  end
end

end
