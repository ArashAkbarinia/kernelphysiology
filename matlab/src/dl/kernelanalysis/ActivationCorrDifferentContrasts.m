function ActivationReport = ActivationCorrDifferentContrasts(net, inim, outdir, SaveImages, layers, ContrastLevels)
%ActivationCorrDifferentContrasts  computing the correlation between
%                                  activities of the kernels of the same
%                                  layer at different levels of contrast

if nargin < 6
  ContrastLevels = [5, 15, 30, 50, 75, 100];
end
if nargin < 5
  % finding out the layers before the max pooling
  layers = MaxInds(net, inf);
  layers = (2:layers(end))';
end
if nargin < 4
  SaveImages = true;
end

ActivationReport = struct();

if ~exist(outdir, 'dir')
  mkdir(outdir);
end

ReportPath = sprintf('%sActivationReport.mat', outdir);
if exist(ReportPath, 'file')
  ActivationReport = load(ReportPath);
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

TopPixels = 0.01;
TopKernels = 0.1;

nLayers = size(layers, 1);

% pixel related matrices
PixelCorrMed = zeros(nContrasts, nContrasts, nLayers);
PixelCorrAvg = zeros(nContrasts, nContrasts, nLayers);
PixelTopDiffMed = zeros(nContrasts, nContrasts, nLayers);
PixelTopDiffAvg = zeros(nContrasts, nContrasts, nLayers);

% kernel related matrices
KernelCorrMed = zeros(nContrasts, nContrasts, nLayers);
KernelCorrAvg = zeros(nContrasts, nContrasts, nLayers);
KernelTopDiffMed = zeros(nContrasts, nContrasts, nLayers);
KernelTopDiffAvg = zeros(nContrasts, nContrasts, nLayers);

for i = 1:nContrasts
  contrast1 = ContrastLevels(i);
  ContrastName1 = sprintf('c%.3u', contrast1);
  for j = i + 1:nContrasts
    contrast2 = ContrastLevels(j);
    ContrastName2 = sprintf('c%.3u', contrast2);
    for l = 1:nLayers
      layer = layers(l, 1);
      LayerName = sprintf('l%.2u', layer);
      
      % retrieving feature matrices
      f1 = ActivationReport.cls.(ContrastName1).(LayerName).features;
      f2 = ActivationReport.cls.(ContrastName2).(LayerName).features;
      
      % computing the correlation along the pixels for a kernel
      [rowsk, colsk, chnsk] = size(f1);
      nPixels = rowsk * colsk;
      PixellTopPercentile = ceil(double(nPixels) .* TopPixels);
      pinds = 1:PixellTopPercentile;
      
      TmpPixCorrAll = zeros(chnsk, 1);
      TmpPixDiffTop = zeros(chnsk, 1);
      for k = 1:chnsk
        TmpPixCorrAll(k, 1) = corr2(f1(:, :, k), f2(:, :, k));
        
        f1k = f1(:, :, k);
        f2k = f2(:, :, k);
        f1f2 = abs(f1k - f2k) ./ max(abs(f1k), abs(f2k));
        f1f2(isnan(f1f2)) = 0;
        f1f2 = sort(f1f2(:), 'descend');
        
        TmpDiffPerc = f1f2(pinds);
        TmpPixDiffTop(k, 1) = mean(TmpDiffPerc);
      end
      
      % replacing the NaNs with 0, otherwise we can take mean or median.
      TmpPixCorrAll(isnan(TmpPixCorrAll)) = 0;
      PixelCorrMed(i, j, l) = median(TmpPixCorrAll);
      PixelCorrAvg(i, j, l) = mean(TmpPixCorrAll);
      PixelTopDiffMed(i, j, l) = median(TmpPixDiffTop);
      PixelTopDiffAvg(i, j, l) = mean(TmpPixDiffTop);
      
      % computing the correlation along the kernels for a pixel
      TmpKerCorrAll = corr3(f1, f2);
      TmpKerCorrAll(isnan(TmpKerCorrAll)) = 0;
      TmpKerDiffTop = PercentageChange3(f1, f2, TopKernels);
      KernelCorrMed(i, j, l) = median(TmpKerCorrAll(:));
      KernelCorrAvg(i, j, l) = mean(TmpKerCorrAll(:));
      KernelTopDiffMed(i, j, l) = median(TmpKerDiffTop(:));
      KernelTopDiffAvg(i, j, l) = mean(TmpKerDiffTop(:));
    end
  end
end

ActivationReport.metrices.PixelCorrMed = PixelCorrMed;
ActivationReport.metrices.PixelCorrAvg = PixelCorrAvg;
ActivationReport.metrices.PixelTopDiffMed = PixelTopDiffMed;
ActivationReport.metrices.PixelTopDiffAvg = PixelTopDiffAvg;
ActivationReport.metrices.KernelCorrMed = KernelCorrMed;
ActivationReport.metrices.KernelCorrAvg = KernelCorrAvg;
ActivationReport.metrices.KernelTopDiffMed = KernelTopDiffMed;
ActivationReport.metrices.KernelTopDiffAvg = KernelTopDiffAvg;

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

save(ReportPath, '-struct', 'ActivationReport');

end

function ActivationReport = ProcessOneContrast(net, layers, inim, outdir, prefix, SaveImages)

[predtype, scores] = classify(net, inim);

ActivationReport.prediction.type = char(predtype);
ActivationReport.prediction.score = max(scores(:));

% going through all the layers and compute their activities
for i = 1:size(layers, 1)
  layer = layers(i, 1);
  
  li = net.Layers(layer);
  
  features = activations(net, inim, li.Name);
  LayerReport.features = features;
  
  LayerName = sprintf('l%.2u', layer);
  ActivationReport.(LayerName) = LayerReport;
  
  if SaveImages
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
