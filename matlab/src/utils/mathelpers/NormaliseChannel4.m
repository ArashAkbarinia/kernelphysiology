function limnor = NormaliseChannel4(limage, a, b, mins, maxs)
%NormaliseChannel4  normalises the range between a and b for a 4D matix.

if nargin < 2
  a = 0;
  b = 1;
end

limnor = zeros(size(limage));

for j = 1:size(limage, 4)
  CurrentImage = limage(:, :, :, j);
  if nargin < 4
    mins = min(min(CurrentImage));
    maxs = max(max(CurrentImage));
  end
  limnor(:, :, :, j) = NormaliseChannel(CurrentImage, a, b, mins, maxs);
end

end
