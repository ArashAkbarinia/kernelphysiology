function FigHand = simplenn_graphical_deconvolution(net, im, im_, dispargs, varargin)
%SIMPLENN_GRAPHICAL_DECONVOLUTION. Deconvoluting activations back to the input layer.
%   HEATMAP = SIMPLENN_GRAPHICAL_DECONVOLUTION(NET, IM, IM_)
%   deconvolutes activations to generate a heatmap of activations,
%   and displays it to the user.
%   NET. The CNN to visualize.
%   IM. The original input image. Needed for user output only.
%   IM_. The adjusted input image used as input to the CNN. If IM_ is a
%        gpuArray computations are run on the GPU.
%
%   SIMPLENN_GRAPHICAL_DECONVOLUTION(...,'OPT',VALUE,...) takes the following options:
%
%   'ReLUPass':: 'Guided Backpropagation'heatmap
%       Sets the method used to deconvolute activations through the ReLU layers.
%       The available methods are 'Backpropagation', 'Deconvnet', and 'Guided
%       Backpropagation'. The default method is 'Guided Backpropagation'
%       since this method usually gives the best results.
%
%   'ConvolutionPass':: 'Standard'
%       Sets the method used to deconvolute activations through the convolution
%       layers. The available methods are 'Relevance Propagation', and
%       'Standard'. The default method is 'Standard', since this
%       method usually gives the best results.
%
%   'MeasureLayer':: Last layer
%       An Int32 specifying the layer from which activations should be
%       deconvoluted back to the input layer. By default the last layer of
%       the network is used.
%
%   'MeasureFilter':: Strongest activated filter
%       An Int32 specifying the filter for which activations should be
%       deconvoluted back to the input layer. By default the strongest
%       activated filter is used.
%
%   ADVANCED USAGE
%
%   Computations are run on the GPU if IM_ is a gpuArray. Otherwise they
%   are run on the CPU.
%

% Copyright (C) 2016 Felix Grün.
% All rights reserved.
%
% This file is part of the FeatureVis library and is made available under
% the terms of the BSD license (see the LICENCE file).


gpuMode = isa(im_, 'gpuArray') ;

% move everything to the GPU
if gpuMode
  net = vl_simplenn_move(net, 'gpu');
else
  net = vl_simplenn_move(net, 'cpu');
end

% forward pass with dropout disabled
res = vl_simplenn(net, im_, [], [], 'Mode', 'test', 'conserveMemory', false) ;

% needed to display the classification result of the network
scores = squeeze(gather(res(end).x)) ;
[classScore, class] = max(scores) ;

heatmap = simplenn_deconvolution(net, im_, varargin{:});

% --- display results ---
FigHand = display(net, im, heatmap, class, classScore, dispargs);

end

function FigHand = display(net, im, heat, class, classScore, dispargs)

% --- output results ---

% filter for the positive activations
heat = heat .* (heat > double(0));

% normalize per pixel over all color channles
for w = 1:size(heat,2)
  for h = 1:size(heat,1)
    heat(h,w,:) = norm(squeeze(heat(h,w,:)));
  end
end
heat = heat / max(heat(:));

% --- "stretch" heatmap to the size of the original image ---

% calculate the resize factors for width and height
fac_h = cast(size(im, 1), 'double') / cast(size(heat, 1), 'double');
fac_w = cast(size(im, 2), 'double') / cast(size(heat, 2), 'double');

% pre-allocate the resized heatmap
im2 = zeros(size(im,1), size(im,2), size(im,3), 'double');

% resize the heatmap
for h = 1:size(heat,1)
  for w = 1:size(heat,2)
    im2(round((h-1)*fac_h)+1:round(h*fac_h), round((w-1)*fac_w)+1:round(w*fac_w), 1) = heat(h,w);
  end
end

% set heatmap values for all color channels
if (size(im,3) > 1)
  for h = 2:size(im,3)
    im2(:,:,h) = im2(:,:,1);
  end
end

% --- output results ---

% added this to handle grey-scaFigHandle images
if size(im, 3) == 1
  im(:, :, 2) = im(:, :, 1);
  im(:, :, 3) = im(:, :, 1);
  
  im2(:, :, 2) = im2(:, :, 1);
  im2(:, :, 3) = im2(:, :, 1);
end

im4 = ind2rgb(double(round(squeeze(im2(:,:,1)) * 255)), jet(255));
im4(:) = cast(round(cast(im4(:), 'double') * 255), 'uint8');

% create overlay
im3 = zeros(size(im,1), size(im,2), size(im,3), 'uint8');
im3(:) = cast(round((cast(im4, 'double') * 0.6) + (cast(im, 'double') * 0.4)), 'uint8');

% display image, heatmap, and overlay
FigHand = figure('Name', 'Simplenn Deconvolution', dispargs{:});
subplot(2, 2, 1); title('Original Image'); imagesc(im); axis('off', 'equal');
subplot(2, 2, 2); title('Black and White Heat Map'); imagesc(im2); axis('off', 'equal');
subplot(2, 2, 3); title('Overlay'); imagesc(im3); axis('off', 'equal');
subplot(2, 2, 4); title('Color Heat Map'); imagesc(im4); axis('off', 'equal');

end
