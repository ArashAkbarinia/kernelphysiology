function ActivationReport = ActivationDifferentContrasts(net, inim, outdir, SaveImages)
%ActivationDifferentContrasts Summary of this function goes here
%   Detailed explanation goes here

if nargin < 4
  SaveImages = true;
end

ActivationReport = struct();

if ~exist(outdir, 'dir')
  mkdir(outdir);
elseif exist(sprintf('%sActivationReport.mat', outdir), 'file')
  fprintf('Skipping %s\n', outdir);
  return;
end

inim = ResizeImageToNet(net, inim);

ContrastLevels = [1, 3, 5, 7, 10, 13, 15, 30, 50, 75, 100];

layers = [2, 4, 7, 9, 12];
nContrasts = numel(ContrastLevels);
for contrast = ContrastLevels
  ContrastedImage = AdjustContrast(inim, contrast / 100);
  ContrastedImage = uint8(ContrastedImage .* 255);
  ContrastName = sprintf('c%.3u', contrast);
  ActivationReport.cls.(ContrastName) = ProcessOneContrast(net, layers, ContrastedImage, outdir, ContrastName, SaveImages);
end

nLayers = numel(layers);

ActivationReport.CompMatrix = zeros(nContrasts, nContrasts, nLayers);
for i = 1:nContrasts
  contrast1 = ContrastLevels(i);
  ContrastName1 = sprintf('c%.3u', contrast1);
  for j = i + 1:nContrasts
    contrast2 = ContrastLevels(j);
    ContrastName2 = sprintf('c%.3u', contrast2);
    for l = 1:nLayers
      layer = layers(l);
      LayerName = sprintf('l%.2u', layer);
      activity1 = ActivationReport.cls.(ContrastName1).(LayerName).top{2};
      activity2 = ActivationReport.cls.(ContrastName2).(LayerName).top{2};
      DiffActivity = activity1 - activity2;
      PerIdenticalNeurons = sum(DiffActivity(:) == 0) / numel(DiffActivity(:));
      ActivationReport.CompMatrix(i, j, l) = PerIdenticalNeurons;
    end
  end
end

% if not save images, remove them from output
if ~SaveImages
  for i = 1:nContrasts
    contrast1 = ContrastLevels(i);
    ContrastName1 = sprintf('c%.3u', contrast1);
    for l = 1:nLayers
      layer = layers(l);
      LayerName = sprintf('l%.2u', layer);
      ActivationReport.cls.(ContrastName1) = rmfield(ActivationReport.cls.(ContrastName1), LayerName);
    end
  end
end

save(sprintf('%sActivationReport.mat', outdir), '-struct', 'ActivationReport');

end

function ActivationReport = ProcessOneContrast(net, layers, inim, outdir, prefix, SaveImages)

[predtype, scores] = classify(net, inim);

ActivationReport.prediction.type = char(predtype);
ActivationReport.prediction.score = max(scores(:));

for layer = layers
  li = net.Layers(layer);
  
  features = activations(net, inim, layer);
  [fsorted, finds] = sort(features, 3, 'descend');
  
  LayerReport.features = features;
  LayerReport.top = {fsorted(:, :, 1), finds(:, :, 1)};
  
  LayerName = sprintf('l%.2u', layer);
  ActivationReport.(LayerName) = LayerReport;
  
  if SaveImages
    nkernels = size(features, 3);
    cmap = DistinguishableColours(nkernels);
    
    rgbim = label2rgb(LayerReport.top{2}, cmap);
    imwrite(rgbim, sprintf('%s%s-l%.2u.png', outdir, prefix, layer));
    
    h = figure('visible', 'off');
    montage(features);
    title(['Layer ', li.Name, ' Activities']);
    saveas(h, sprintf('%s%s-montage%.2u.png', outdir, prefix, layer));
    close(h);
  end
end

end
