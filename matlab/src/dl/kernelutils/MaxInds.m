function MaxLayers = MaxInds(net, topn)
%MaxInds  Finds the indices of the first max-pooling layers.
%
% inputs
%  net   the network.
%  topn  the number of layers to be returned, by default all.
%
% outputs
%  MaxLayers  indices of the first N max pooling layers.
%

if nargin < 2
  topn = inf;
end

MaxLayers = [];
for i = 1:numel(net.Layers)
  if isa(net.Layers(i), 'nnet.cnn.layer.MaxPooling2DLayer')
    MaxLayers(end + 1, 1) = i; %#ok
    if numel(MaxLayers) == topn
      return;
    end
  end
end

end

