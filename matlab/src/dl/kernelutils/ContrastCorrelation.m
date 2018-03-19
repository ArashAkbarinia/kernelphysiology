function CorrMat = ContrastCorrelation( net, layers )
%ContrastCorrelation Summary of this function goes here
%   Detailed explanation goes here

% TODO: implement proper layer handling.
if nargin < 2
  layers = 1;
end

% NOTICE: deciding according to the field 'params' might not be the best.
if isfield(net, 'params')
  CorrMat = ContrastCorrelationDagNN(net, layers);
else
  CorrMat = ContrastCorrelationSimple(net, layers);
end

end

function CorrMat = ContrastCorrelationDagNN( net, layers )

[rows, cols, chns, kers] = size(net.params(layers).value);

CorrMat = ones(kers, kers, chns);

for c = 1:chns
  for k1 = 1:kers - 1
    for k2 = k1 + 1:kers
      ker1 = net.params(layers).value(:, :, c, k1);
      ker2 = net.params(layers).value(:, :, c, k2);
      
      CorrMat(k1, k2, c) = corr2(ker1, ker2);
      
      CorrMat(k2, k1, c) = CorrMat(k1, k2, c);
    end
  end
end

end

function CorrMat = ContrastCorrelationSimple( net, layers )

[rows, cols, chns, kers] = size(net.layers{layers}.weights{1, 1});

CorrMat = ones(kers, kers, chns);

for c = 1:chns
  for k1 = 1:kers - 1
    for k2 = k1 + 1:kers
      ker1 = net.layers{layers}.weights{1, 1}(:, :, c, k1);
      ker2 = net.layers{layers}.weights{1, 1}(:, :, c, k2);
      
      %       ker1 = NormaliseChannel(ker1);
      %       ker2 = NormaliseChannel(ker2);
      %
      %       ker1(ker1 < 0) = -1;
      %       ker1(ker1 > 0) = +1;
      %       ker2(ker2 < 0) = -1;
      %       ker2(ker2 > 0) = +1;
      
      %       CorrMat(k1, k2, c) = corr(ker1(:), ker2(:));
      CorrMat(k1, k2, c) = corr2(ker1, ker2);
      %       CorrMat(k1, k2, c) = corr(real(eig(ker1)), real(eig(ker2)));
      %       CorrMat(k1, k2, c) = real(corr(eig(ker1), eig(ker2)));
      
      CorrMat(k2, k1, c) = CorrMat(k1, k2, c);
    end
  end
end

% IdentityImage = eye(kers);
% IdentityImage = repmat(IdentityImage, [1, 1, chns]);
% IdentityImage = logical(IdentityImage);
%
% CorrMat(IdentityImage) = inf;

end
