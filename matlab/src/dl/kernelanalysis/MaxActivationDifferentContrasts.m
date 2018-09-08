function ActivationReport = MaxActivationDifferentContrasts(net, inim, outdir, SaveImages, layers, ContrastLevels)
%MaxActivationDifferentContrasts  computing the activity of layers before
%                                 max pooling.

if nargin < 6
  ContrastLevels = [5, 15, 30, 50, 75, 100];
end
if nargin < 5
  % finding out the layers before the max pooling
  layers = BeforeMaxInds(net, 5);
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

nLayers = size(layers, 1);
binranges = cell(nLayers, 1);
nEdges = 100;
HistPercentage = 0.75;
% computing the histograms in a range specific to a layer
for i = nContrasts:-1:1
  contrast = ContrastLevels(i);
  ContrastName = sprintf('c%.3u', contrast);
  for l = 1:nLayers
    layer = layers(l, 1);
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
      % when computing the histogram for the kernel we don't consider the
      % previous ranges, we basically are interested to check whether the
      % shape remains the same
      histvals = histcounts(CurrentKernel(:), nEdges);
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
      layer = layers(l, 1);
      layermax = layers(l, 2);
      LayerName = sprintf('l%.2u', layer);
      activity1 = ActivationReport.cls.(ContrastName1).(LayerName).top;
      activity2 = ActivationReport.cls.(ContrastName2).(LayerName).top;
      DiffActivity = activity1 & activity2;
      % computing regional trues according to max pooling stride
      DiffActivity = RegionalTrues(DiffActivity, net.Layers(layermax));
      
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

for i = 1:size(layers, 1)
  layer = layers(i, 1);
  % the max layer
  limax = net.Layers(layers(i, 2));
  
  % retreiving the layer before the max pooling
  li = net.Layers(layer);
  
  features = activations(net, inim, li.Name);
  FeaturesMax = MaxPooling(features, limax);
  features_max = activations(net, inim, limax.Name);
  
  if sum(features_max(:) - FeaturesMax(:)) ~= 0
    disp(num2str(sum(features_max(:) - FeaturesMax(:))))
  end
  
  % to make it the same size as the one before the max pooling
  FeaturesMax = repelem(FeaturesMax, limax.Stride(1), limax.Stride(2), 1);
  
  % NOTE: when pooling size and stride mismatch
  [RowsMax, ColsMax, ~] = size(FeaturesMax);
  if ColsMax ~= size(features, 2)
    FeaturesMax(:, ColsMax + 1, :) = FeaturesMax(:, ColsMax, :);
  end
  if RowsMax ~= size(features, 1)
    FeaturesMax(RowsMax + 1, :, :) = FeaturesMax(RowsMax, :, :);
  end
  
  % finding out which pixels have been selected by max pooling
  RegionalMaxs = features == FeaturesMax;
  
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

function TrueImage = RegionalTrues(ComparisonMatrix, limax)

TrueImage = RegionalPooling(ComparisonMatrix, limax, 'or');

end

function TrueImage = MaxPooling(FeatureMatrix, limax)

TrueImage = RegionalPooling(FeatureMatrix, limax, 'max');

end

function TrueImage = RegionalPooling(FeatureMatrix, limax, OperationType)

PoolSize = limax.PoolSize;
stride = limax.Stride;
padding = limax.PaddingSize;

[rows, cols, chns] = size(FeatureMatrix);

if padding(1) > 0
  FeatureMatrix = cat(1, FeatureMatrix(1:padding(1), :, :), FeatureMatrix);
end
if padding(2) > 0
  FeatureMatrix = cat(1, FeatureMatrix, FeatureMatrix(rows - padding(2) + 1:rows, :, :));
end
if padding(3) > 0
  FeatureMatrix = cat(2, FeatureMatrix(:, 1:padding(3), :), FeatureMatrix);
end
if padding(4) > 0
  FeatureMatrix = cat(2, FeatureMatrix, FeatureMatrix(:, cols - padding(4) + 1:cols, :));
end

% NOTE: floor is used for when stride and max pooling size is different
RowsPool = floor(rows / stride(1));
ColsPool = floor(cols / stride(2));
TrueImage = zeros(RowsPool, ColsPool, chns);

for i = 1:PoolSize(1)
  for j = 1:PoolSize(2)
    maxim = FeatureMatrix(i:stride(1):end, j:stride(2):end, :);
    [RowsMax, ColsMax, ~] = size(maxim);
    ColDiff = ColsMax - ColsPool;
    if ColDiff > 0
      maxim = maxim(:, 1:ColsMax - ColDiff, :);
    elseif ColDiff < 0
      ColDiff = abs(ColDiff);
      maxim(1:RowsMax, ColsMax + 1:ColsMax + ColDiff, :) = FeatureMatrix(i:stride(1):end, cols - ColDiff + 1:cols, :);
    end
    RowDiff = RowsMax - RowsPool;
    if RowDiff > 0
      maxim = maxim(1:RowsMax - RowDiff, :, :);
    elseif RowDiff < 0
      RowDiff = abs(RowDiff);
      maxim(RowsMax + 1:RowsMax + RowDiff, 1:ColsMax, :) = FeatureMatrix(rows - RowDiff + 1:rows, j:stride(2):end, :);
    end
    if strcmpi(OperationType, 'max')
      TrueImage = max(TrueImage, maxim);
    elseif strcmpi(OperationType, 'or')
      TrueImage = TrueImage | maxim;
    end
  end
end

end
