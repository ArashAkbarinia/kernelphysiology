function KernelsComparisonMat = KernelsComparison(AllKernels, JoinedChannels)
%KernelsComparison Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
  JoinedChannels = true;
end

if JoinedChannels
  KernelsComparisonMat = DoJoined(AllKernels);
else
  KernelsComparisonMat = DoSeparated(AllKernels);
end

end

function KernelsComparisonMat = DoJoined(AllKernels)

AllKernels = double(AllKernels);
[~, ~, ~, kers] = size(AllKernels);
KernelsComparisonMat = ones(kers, kers);

for k1 = 1:kers - 1
  for k2 = k1 + 1:kers
    ker1 = AllKernels(:, :, :, k1);
    ker2 = AllKernels(:, :, :, k2);
    
    KernelsComparisonMat(k1, k2) = corr3c(ker1, ker2);
    
    KernelsComparisonMat(k2, k1) = KernelsComparisonMat(k1, k2);
  end
end

end

function KernelsComparisonMat = DoSeparated(AllKernels)

[~, ~, chns, kers] = size(AllKernels);
KernelsComparisonMat = ones(kers, kers, chns);

for c = 1:chns
  for k1 = 1:kers - 1
    for k2 = k1 + 1:kers
      ker1 = AllKernels(:, :, c, k1);
      ker2 = AllKernels(:, :, c, k2);
      
      KernelsComparisonMat(k1, k2, c) = corr2(ker1, ker2);
      
      KernelsComparisonMat(k2, k1, c) = KernelsComparisonMat(k1, k2, c);
    end
  end
end

end
