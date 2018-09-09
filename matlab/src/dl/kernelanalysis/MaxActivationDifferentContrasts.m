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
      
      % size of the pooled matrix is size of all the regions devided by the
      % pooling size
      [rowsk, colsk, chnsk] = size(DiffActivity);
      rowsk = rowsk / net.Layers(layermax).PoolSize(1);
      colsk = colsk / net.Layers(layermax).PoolSize(2);
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
  
  RegionalMaxs = FindRegionalMaxes(FeaturesMax, features, limax);
  
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

TrueImage = FindAnyMaxRegion(ComparisonMatrix, limax);

end

function TrueImage = MaxPooling(FeatureMatrix, limax)

TrueImage = RegionalPooling(FeatureMatrix, limax);

end

function TrueImage = RegionalPooling(FeatureMatrix, limax)

PoolSize = limax.PoolSize;
stride = limax.Stride;
padding = limax.PaddingSize;

[rows, cols, chns] = size(FeatureMatrix);

FeatureMatrix = AddPadding(FeatureMatrix, padding);

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
    TrueImage = max(TrueImage, maxim);
  end
end

end

function RegionalMax = FindRegionalMaxes(FeaturesMax, FeatureMatrix, limax)

PoolSize = limax.PoolSize;
stride = limax.Stride;
padding = limax.PaddingSize;

[rows, cols, chns] = size(FeatureMatrix);

FeatureMatrix = AddPadding(FeatureMatrix, padding);

[RowsPool, ColsPool, ~] = size(FeaturesMax);

% a logical matrix that presents which pixels have been the most activated
% one within their region
RegionalMax = false(RowsPool * PoolSize(1), ColsPool * PoolSize(2), chns);

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
    CurrentMax = FeaturesMax == maxim;
    
    RegionalMax(i:PoolSize(1):end, j:PoolSize(2):end, :) = RegionalMax(i:PoolSize(1):end, j:PoolSize(2):end, :) | CurrentMax;
  end
end

end

function RegionalMax = FindAnyMaxRegion(FeaturesMax, limax)

PoolSize = limax.PoolSize;

[rows, cols, chns] = size(FeaturesMax);

% a logical matrix that presents which pixels have been the most activated
% one within their region
RegionalMax = false(rows / PoolSize(1), cols / PoolSize(2), chns);

for i = 1:PoolSize(1)
  for j = 1:PoolSize(2)
    maxim = FeaturesMax(i:PoolSize(1):end, j:PoolSize(2):end, :);
    RegionalMax = RegionalMax | maxim;
  end
end

end

function inmat = AddPadding(inmat, padding)

[rows, cols, ~] = size(inmat);

if padding(1) > 0
  inmat = cat(1, inmat(1:padding(1), :, :), inmat);
end
if padding(2) > 0
  inmat = cat(1, inmat, inmat(rows - padding(2) + 1:rows, :, :));
end
if padding(3) > 0
  inmat = cat(2, inmat(:, 1:padding(3), :), inmat);
end
if padding(4) > 0
  inmat = cat(2, inmat, inmat(:, cols - padding(4) + 1:cols, :));
end

end
