function ConvLayers = ConvInds(net, topn)
%ConvInds  Finds the indices of the first few convlutional layers.
%
% inputs
%  net   the network.
%  topn  the number of layers to be returned, by default all.
%
% outputs
%  ConvLayers  indices of the first N convolutional layers.
%

if nargin < 2
  topn = inf;
end

ConvLayers = [];
for i = 1:numel(net.Layers)
  if isa(net.Layers(i), 'nnet.cnn.layer.Convolution2DLayer')
    ConvLayers(end + 1, 1) = i; %#ok
    if numel(ConvLayers) == topn
      return;
    end
  end
end

end
