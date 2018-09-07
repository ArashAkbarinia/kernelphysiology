function BeforeMaxLayers = BeforeMaxInds(net, topn)
%BeforeMaxInds  Finds the indices of the layers before max-pooling.
%
% inputs
%  net   the network.
%  topn  the number of layers to be returned, by default all.
%
% outputs
%  BeforeMaxLayers  indices of the layers before each max pooling.
%

if nargin < 2
  topn = inf;
end

BeforeMaxLayers = [];
for i = 1:numel(net.Layers)
  if isa(net.Layers(i), 'nnet.cnn.layer.MaxPooling2DLayer')
    if isa(net, 'DAGNetwork')
      ConnectionId = strcmpi(net.Connections.Destination, net.Layers(i).Name);
      LayerId = LayerName2Id(net, net.Connections.Source(ConnectionId));
      BeforeMaxLayers(end + 1, :) = [LayerId, i]; %#ok
    elseif isa(net, 'SeriesNetwork')
      BeforeMaxLayers(end + 1, :) = [i - 1, i]; %#ok
    else
      error('Unsopported network architecture!');
    end
    
    if numel(BeforeMaxLayers) == topn
      return;
    end
  end
end

end
