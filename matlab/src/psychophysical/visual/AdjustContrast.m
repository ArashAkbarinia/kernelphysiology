function OutImage = AdjustContrast(InputImage, ContrastLevel, PixelDeviation)
%AdjustContrast  it scales the image to a certain contrast level in [0, 1].
%   https://github.com/rgeirhos/object-recognition/blob/master/code/image-manipulation.py
% contrast_level: a scalar in [0, 1]; with 1 -> full contrast

if nargin < 3
  PixelDeviation = 0;
end

InputImage = im2double(InputImage);

assert(ContrastLevel >= 0.0, 'ContrastLevel too low.');
assert(ContrastLevel <= 1.0, 'ContrastLevel too high.');

MinContrast = ContrastLevel - PixelDeviation;
MaxContrast = ContrastLevel + PixelDeviation;

PixelDeviation = (MaxContrast - MinContrast) .* rand(size(InputImage)) + MinContrast;
ContrastLevel = ones(size(InputImage)) .* PixelDeviation;

OutImage = (1 - ContrastLevel) ./ 2.0 + InputImage .* ContrastLevel;

end
