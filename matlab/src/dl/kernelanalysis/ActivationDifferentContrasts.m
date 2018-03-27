function ActivationReport = ActivationDifferentContrasts(net, inim)
%ActivationDifferentContrasts Summary of this function goes here
%   Detailed explanation goes here

outdir = '/home/arash/Software/repositories/kernelphysiology/analysis/kernelsactivity/t01/';
mkdir(outdir);

imsize = net.Layers(1).InputSize;

% convert it to the network input size
inim = imresize(inim, imsize(1:2));

ContrastLevels = [1, 3, 5, 7, 10, 13, 15, 30, 50, 75, 100] ./ 100;

layers = [2, 4, 7, 9, 12];
nContrasts = numel(ContrastLevels);
for contrast = ContrastLevels
  ContrastedImage = AdjustContrast(inim, contrast);
  ContrastedImage = uint8(ContrastedImage .* 255);
  ContrastName = sprintf('c%.3d', contrast .* 100);
  ActivationReport.(ContrastName) = ProcessOneContrast(net, layers, ContrastedImage, outdir, ContrastName);
end

nLayers = numel(layers);

ActivationReport.CompMatrix = zeros(nContrasts, nContrasts, nLayers);
for i = 1:nContrasts
  contrast1 = ContrastLevels(i);
  ContrastName1 = sprintf('c%.3d', contrast1 .* 100);
  for j = i + 1:nContrasts
    contrast2 = ContrastLevels(j);
    ContrastName2 = sprintf('c%.3d', contrast2 .* 100);
    for l = 1:nLayers
      layer = layers(l);
      LayerName = sprintf('l%.2d', layer);
      activity1 = ActivationReport.(ContrastName1).(LayerName).top{2};
      activity2 = ActivationReport.(ContrastName2).(LayerName).top{2};
      DiffActivity = activity1 - activity2;
      PerIdenticalNeurons = sum(DiffActivity(:) == 0) / numel(DiffActivity(:));
      ActivationReport.CompMatrix(i, j, l) = PerIdenticalNeurons;
    end
  end
end

save(sprintf('%sActivationReport.mat', outdir), '-struct', 'ActivationReport');

end

function ActivationReport = ProcessOneContrast(net, layers, inim, outdir, prefix)

[predtype, scores] = classify(net, inim);
disp(predtype); disp(scores(149))

for layer = layers
  li = net.Layers(layer);
  
  features = activations(net, inim, layer);
  [fsorted, finds] = sort(features, 3, 'descend');
  
  LayerReport.features = features;
  LayerReport.top = {fsorted(:, :, 1), finds(:, :, 1)};
  
  LayerName = sprintf('l%.2d', layer);
  ActivationReport.(LayerName) = LayerReport;
  
  nkernels = size(features, 3);
  cmap = distinguishable_colors(nkernels);
  
  rgbim = label2rgb(LayerReport.top{2}, cmap);
  imwrite(rgbim, sprintf('%s%s-l%.2d.png', outdir, prefix, layer));
  
  h = figure('visible', 'off');
  montage(features);
  title(['Layer ', li.Name, ' Activities']);
  saveas(h, sprintf('%s%s-montage%.2d.png', outdir, prefix, layer));
  close(h);
end

end
