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

nLayers = size(layers, 1);

PixelCorrMed = zeros(nContrasts, nContrasts, nLayers);
PixelCorrAvg = zeros(nContrasts, nContrasts, nLayers);
KernelCorrMed = zeros(nContrasts, nContrasts, nLayers);
KernelCorrAvg = zeros(nContrasts, nContrasts, nLayers);
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
      chnsk = size(f1, 3);
      KernelCorrMatrix = zeros(chnsk, 1);
      for k = 1:chnsk
        KernelCorrMatrix(k, 1) = corr2(f1(:, :, k), f2(:, :, k));
      end
      % replacing the NaNs with 0, otherwise we can take mean or median.
      KernelCorrMatrix(isnan(KernelCorrMatrix)) = 0;
      PixelCorrMed(i, j, l) = median(KernelCorrMatrix);
      PixelCorrAvg(i, j, l) = mean(KernelCorrMatrix);
      
      % computing the correlation along the kernels for a pixel
      PixelCorrMatrix = corr3(f1, f2);
      PixelCorrMatrix(isnan(PixelCorrMatrix)) = 0;
      KernelCorrMed(i, j, l) = median(PixelCorrMatrix(:));
      KernelCorrAvg(i, j, l) = mean(PixelCorrMatrix(:));
    end
  end
end

ActivationReport.metrices.PixelCorrMed = PixelCorrMed;
ActivationReport.metrices.PixelCorrAvg = PixelCorrAvg;
ActivationReport.metrices.KernelCorrMed = KernelCorrMed;
ActivationReport.metrices.KernelCorrAvg = KernelCorrAvg;

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

for i = 1:size(layers, 1)
  layer = layers(i, 1);
  
  % retreiving the layer before the max pooling
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
