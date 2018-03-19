function cimg = StandardImage( cimg, npixels )
%StandardImage Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
  npixels = size(cimg, 1) * size(cimg, 2) * size(cimg, 3);
end

cimg = double(cimg) ./ 255;

AvgVal = mean(cimg(:));
StdVal = std(cimg(:));
AdjStd = max(StdVal, 1 / sqrt(npixels));

cimg = (cimg - AvgVal) ./ AdjStd;

cimg = single(cimg);

end
