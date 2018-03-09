function [  ] = FeatureVisAllKernels( net, im, OutFolder )
%FeatureVisAllKernels Summary of this function goes here
%   Detailed explanation goes here

stdim = rgb2gray(im);

stdim = StandardImage(stdim);

nlayers = numel(net.layers);

dispargs = {'visible', 'off'};

for i = 1:nlayers
  LayerName = net.layers{i}.type;
  CurrentOutFolder = sprintf('%s/layer%d-%s', OutFolder, i, LayerName);
  mkdir(CurrentOutFolder);
  if strcmpi(LayerName, 'conv')
    nneurons = size(net.layers{i}.weights{1}, 4);
  end
  for j = 1:nneurons
    FigHand = graphical_deconvolution(net, im, stdim, dispargs, 'MeasureLayer', i, 'MeasureFilter', j);
    SaveFileName  = sprintf('%s/neuron%d.jpg', CurrentOutFolder, j);
    saveas(FigHand, SaveFileName);
    close(FigHand);
  end
end

end
