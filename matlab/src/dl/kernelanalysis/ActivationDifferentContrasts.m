function ActivationReport = ActivationDifferentContrasts(net, inim, outdir, SaveImages, layers, ContrastLevels)
%ActivationDifferentContrasts Summary of this function goes here
%   Detailed explanation goes here

if nargin < 6
  ContrastLevels = [1, 3, 5, 7, 10, 13, 15, 30, 50, 75, 100];
end
if nargin < 5
  layers = ConvInds(net, 5);
end
if nargin < 4
  SaveImages = true;
end

ActivationReport = struct();

if ~exist(outdir, 'dir')
  mkdir(outdir);
elseif exist(sprintf('%sActivationReport.mat', outdir), 'file')
  ActivationReport = load(sprintf('%sActivationReport.mat', outdir));
  return;
end

inim = ResizeImageToNet(net, inim);

nContrasts = numel(ContrastLevels);
for contrast = ContrastLevels
  ContrastedImage = AdjustContrast(inim, contrast / 100);
  ContrastedImage = uint8(ContrastedImage .* 255);
  ContrastName = sprintf('c%.3u', contrast);
  ActivationReport.cls.(ContrastName) = ProcessOneContrast(net, layers, ContrastedImage, outdir, ContrastName, SaveImages);
end

nLayers = numel(layers);
binranges = cell(nLayers, 1);
nEdges = 100;
HistPercentage = 0.75;
% computing the histograms in a range specific to a layer
for i = nContrasts:-1:1
  contrast = ContrastLevels(i);
  ContrastName = sprintf('c%.3u', contrast);
  for l = 1:nLayers
    layer = layers(l);
    LayerName = sprintf('l%.2u', layer);
    features = ActivationReport.cls.(ContrastName).(LayerName).features;
    if i == nContrasts
      binranges{l} = linspace(HistPercentage * min(features(:)), HistPercentage * max(features(:)), nEdges - 1);
      % NOTE: histcounts has a bug that donese't take the values outside of
      % the rane.
      binranges{l} = [-inf, binranges{l}, +inf];
    end
    histvals = histc(features, binranges{l}, 3);
    histvals = histvals ./ sum(histvals, 3);
    ActivationReport.cls.(ContrastName).(LayerName).histogram = histvals;
  end
end

ActivationReport.CompMatrix = zeros(nContrasts, nContrasts, nLayers);
ActivationReport.CompMatrixHist = zeros(nContrasts, nContrasts, nLayers);
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
      
      % histogram comparisons
      hist1 = ActivationReport.cls.(ContrastName1).(LayerName).histogram;
      hist2 = ActivationReport.cls.(ContrastName2).(LayerName).histogram;
      
      % Euclidean distance between histograms
      HistDiff = (hist1 - hist2) .^ 2;
      HistDiff = sqrt(sum(HistDiff, 3));
      ActivationReport.CompMatrixHist(i, j, l) = mean(HistDiff(:));
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
  
  features = activations(net, inim, net.Layers(layer).Name);
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
