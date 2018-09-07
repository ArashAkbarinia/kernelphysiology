function ActivationReport = MaxActivationDifferentContrasts(net, inim, outdir, SaveImages, layers, ContrastLevels)
%MaxActivationDifferentContrasts  computing the activity of layers before
%                                 max pooling.

if nargin < 6
  ContrastLevels = [5, 15, 30, 50, 75, 100];
end
if nargin < 5
  % finding out the layers before the max pooling
  layers = MaxInds(net, 5) - 1;
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

% identical across different level of contrasts.
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
    [~, ~, chnsk] = size(features);
    KernelHistograms = zeros(chnsk, nEdges);
    for k = 1:chnsk
      CurrentKernel = features(:, :, k);
      histvals = histcounts(CurrentKernel(:), binranges{l});
      KernelHistograms(k, :) = histvals ./ sum(histvals);
    end
    ActivationReport.cls.(ContrastName).(LayerName).histogram = KernelHistograms;
  end
end

ActivationReport.CompMatrix = zeros(nContrasts, nContrasts, nLayers);
ActivationReport.CompMatrixHist = zeros(nContrasts, nContrasts, nLayers);
% storign the values per each kernel is too heavy for the memory
if SaveImages
  ActivationReport.KernelMatrix = cell(nContrasts, nContrasts, nLayers);
  ActivationReport.KernelMatrixHist = cell(nContrasts, nContrasts, nLayers);
end
for i = 1:nContrasts
  contrast1 = ContrastLevels(i);
  ContrastName1 = sprintf('c%.3u', contrast1);
  for j = i + 1:nContrasts
    contrast2 = ContrastLevels(j);
    ContrastName2 = sprintf('c%.3u', contrast2);
    for l = 1:nLayers
      layer = layers(l);
      LayerName = sprintf('l%.2u', layer);
      activity1 = ActivationReport.cls.(ContrastName1).(LayerName).top;
      activity2 = ActivationReport.cls.(ContrastName2).(LayerName).top;
      DiffActivity = activity1 & activity2;
      % computing regional trues according to max pooling stride
      DiffActivity = RegionalTrues(DiffActivity, net.Layers(layer + 1).PoolSize, net.Layers(layer + 1).Stride);
      
      [rowsk, colsk, chnsk] = size(DiffActivity);
      nPixels = rowsk * colsk;
      KernelMatrix = zeros(chnsk);
      KernelMatrixSum = 0;
      for k = 1:chnsk
        CurrentKernel = DiffActivity(:, :, k);
        PerIdenticalNeurons = sum(CurrentKernel(:)) / nPixels;
        KernelMatrix(k) = PerIdenticalNeurons;
        KernelMatrixSum = KernelMatrixSum + PerIdenticalNeurons;
      end
      if SaveImages
        ActivationReport.KernelMatrix{i, j, l} = KernelMatrix;
      end
      ActivationReport.CompMatrix(i, j, l) = KernelMatrixSum / chnsk;
      
      % histogram comparisons
      hist1 = ActivationReport.cls.(ContrastName1).(LayerName).histogram;
      hist2 = ActivationReport.cls.(ContrastName2).(LayerName).histogram;
      
      % Euclidean distance between histograms
      HistDiff = (hist1 - hist2) .^ 2;
      HistDiff = sqrt(sum(HistDiff, 2));
      
      if SaveImages
        ActivationReport.KernelMatrixHist{i, j, l} = HistDiff;
      end
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
  % the max layer
  li_max = net.Layers(layer + 1);
  
  % retreiving the layer before the max pooling
  li = net.Layers(layer);
  
  features = activations(net, inim, li.Name);
  features_max = activations(net, inim, li_max.Name);
  
  % to make it the same size as the one before the max pooling
  features_max = repelem(features_max, li_max.Stride(1), li_max.Stride(2), 1);
  
  % NOTE: when pooling size and stride mismatch
  [rows_max, cols_max, ~] = size(features_max);
  if cols_max ~= size(features, 2)
    features_max(:, cols_max + 1, :) = features_max(:, cols_max, :);
  end
  if rows_max ~= size(features, 1)
    features_max(rows_max + 1, :, :) = features_max(rows_max, :, :);
  end
  
  % finding out which pixels have been selected by max pooling
  RegionalMaxs = features == features_max;
  
  LayerReport.features = features;
  LayerReport.top = RegionalMaxs;
  
  LayerName = sprintf('l%.2u', layer);
  ActivationReport.(LayerName) = LayerReport;
  
  if SaveImages
    h = figure('visible', 'off');
    minval = min(min(LayerReport.top));
    maxval = max(max(LayerReport.top));
    montage(NormaliseChannel(LayerReport.top, 0, 1, minval, maxval));
    title(['Layer ', li.Name, ' Regional Max']);
    saveas(h, sprintf('%s%s-regmax%.2u.png', outdir, prefix, layer));
    close(h);
    
    h = figure('visible', 'off');
    minval = min(min(features));
    maxval = max(max(features));
    montage(NormaliseChannel(features, 0, 1, minval, maxval));
    title(['Layer ', li.Name, ' Activities']);
    saveas(h, sprintf('%s%s-montage%.2u.png', outdir, prefix, layer));
    close(h);
  end
end

end

function TrueImage = RegionalTrues(ComparisonMatrix, PoolSize, stride)

[rows, cols, chns] = size(ComparisonMatrix);
% NOTE: floor is used for when stride and max pooling size is different
rows = floor(rows / stride(1));
cols = floor(cols / stride(2));
TrueImage = false(rows, cols, chns);

% going only till one minus the rows and cols, because pooling doesn't take
% them
for i = 1:stride(1):rows - stride(1)
  erow = i + PoolSize(1) - 1;
  for j = 1:stride(2):cols - stride(2)
    ecol = j + PoolSize(2) - 1;
    tmp = ComparisonMatrix(i:erow, j:ecol, :);
    TrueImage(i:erow, j:ecol, :) = TrueImage(i:erow, j:ecol, :) | tmp;
  end
end

end
