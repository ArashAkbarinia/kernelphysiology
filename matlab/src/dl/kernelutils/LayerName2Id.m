function LayerId = LayerName2Id(net, LayerName)
%LayerName2Id  Finding the index of layer id based on its name.
%
% inputs
%   net        the network.
%   LayerName  the name of the layer.
%
% outputs
%   LayerId  the ID of the layer corresponding to the name.
%

for LayerId = 1:numel(net.Layers)
  if strcmpi(net.Layers(LayerId).Name, LayerName)
    return;
  end
end

% in case we don't find it.
LayerId = -1;

end
