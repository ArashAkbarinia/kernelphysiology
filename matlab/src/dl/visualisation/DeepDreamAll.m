function [] = DeepDreamAll(net, netname)

nlayers = numel(net.Layers);

convname = cell(2, 1);
convname{1} = 'nnet.cnn.layer.Convolution2DLayer';
convname{2} = 'nnet.cnn.layer.FullyConnectedLayer';
%convname{3} = 'nnet.cnn.layer.ClassificationOutputLayer';

for i = 1:nlayers
  fprintf('Processing layer %u\n', i);
  li = net.Layers(i);
  
  if i < 9
    BatchGap = 256;
  elseif i < 15
    BatchGap = 81;
  elseif i < 27
    BatchGap = 36;
  elseif i < 33
    BatchGap = 15;
  elseif i < 39
    BatchGap = 10;
  else
    BatchGap = 5;
  end
  
  if isa(li, convname{1}) || isa(li, convname{2})
    outdir = sprintf('/home/arash/Software/repositories/kernelphysiology/analysis/kernelsvisualised/%s/%s/', netname, li.Name);
    mkdir(outdir);
    
    if isa(li, convname{1})
      DeepDreamLayer(net, i, BatchGap, li.NumFilters, outdir);
    else
      DeepDreamLayer(net, i, BatchGap, li.OutputSize, outdir);
    end
  end
end

end

function [] = DeepDreamLayer(net, layer, BatchGap, maxfilters, outdir)

li = net.Layers(layer);

for k = 0:BatchGap:maxfilters - 1
  channels = 1:BatchGap;
  BatchSize = k;
  channels = channels + BatchSize;
  channels = channels(channels <= maxfilters);
  if exist(sprintf('%smontage%.2u.png', outdir, (BatchSize / BatchGap) + 1), 'file')
    continue;
  end
  
  limage = deepDreamImage(net, layer, channels, 'PyramidLevels', 1, 'ExecutionEnvironment', 'gpu', 'NumIterations', 50);
  
  for j = 1:size(limage, 4)
    CurrentImage = limage(:, :, :, j);
    imwrite(uint8(CurrentImage .* 255), sprintf('%s%.4u.png', outdir, j + BatchSize));
  end
  
  h = figure('visible', 'off');
  montage(limage);
  title(['Layer ', li.Name, ' Features']);
  saveas(h, sprintf('%smontage%.2u.png', outdir, (BatchSize / BatchGap) + 1));
  close(h);
  
  limnor = NormaliseChannel4(limage);
  
  h = figure('visible', 'off');
  montage(limnor);
  title(['Layer ', li.Name, ' Features']);
  saveas(h, sprintf('%smontage-normalised%.2u.png', outdir, (BatchSize / BatchGap) + 1));
  close(h);
end

end
