function inim = ResizeImageToNet(net, inim, docrop)
%ResizeImageToNet  resize the input inmage to the input of network.
%
% inputs
%  net     the DNN network.
%  inim    the image to be resized.
%  docrop  crop the image to the central part.
%
% outputs
%  inim  the resized image prepared to be tested with the network.
%

if nargin < 3
  docrop = true;
end

imsize = net.Layers(1).InputSize;

[rows, cols, chns] = size(inim);

if imsize(1) ~= rows || imsize(2) ~= cols
  if docrop && rows ~= cols
    inim = CropCentreSquareImage(inim);
  end
  
  % convert it to the network input size
  inim = imresize(inim, imsize(1:2));
end

if chns == 1 && imsize(3) == 3
  inim(:, :, 2) = inim(:, :, 1);
  inim(:, :, 3) = inim(:, :, 1);
end

end
