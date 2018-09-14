function pchange = PercentageChange3(mat1, mat2, TopPercentile)
%PercentageChange3  Computes the percentage of changes betwenn two matrices
%
% inputs
%   mat1  the input matrix 1
%   mat2  the input matrix 2
%   TopPercentile  which percentile to eb considered for computation
%
% outputs
%   pchange  percentage of change at every pixel.
%

pchange = abs(mat1 - mat2) ./ max(abs(mat1), abs(mat2));
pchange(isnan(pchange)) = 0;

if nargin > 2
  [~, ~, chnsk] = size(mat1);
  KernelTopPercentile = ceil(double(chnsk) .* TopPercentile);
  kinds = 1:KernelTopPercentile;
  
  pchange = sort(pchange, 3, 'descend');
  pchange = pchange(:, :, kinds);
end

pchange = mean(pchange, 3);

end
