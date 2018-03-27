function OutImage = AdjustContrast(InputImage, ContrastLevel)
%AdjustContrast  it scales the image to a certain contrast level in [0, 1].
%   https://github.com/rgeirhos/object-recognition/blob/master/code/image-manipulation.py
% contrast_level: a scalar in [0, 1]; with 1 -> full contrast

InputImage = im2double(InputImage);

assert(ContrastLevel >= 0.0, 'ContrastLevel too low.');
assert(ContrastLevel <= 1.0, 'ContrastLevel too high.');

OutImage = (1 - ContrastLevel) ./ 2.0 + InputImage .* ContrastLevel;

end
