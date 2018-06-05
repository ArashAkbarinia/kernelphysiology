function I = DeepDream(net, layer, channels, varargin)
%deepDreamImage Visualize network features using Deep Dream.
%
%  Deep Dream is a deep learning feature visualization technique that
%  synthesizes images that strongly activate network layers. Visualizing
%  these images highlight the image features learned by a network. These
%  images are useful for understanding and diagnosing network behavior.
%
%  I = deepDreamImage(network, layer, channel) returns an image
%  that strongly activates the channels of a layer within the network. The
%  input network must be a SeriesNetwork object. layer must be a numeric
%  index or a character vector corresponding to one of the layers in
%  network.Layers. channel must be a scalar or a vector of channel indices.
%  When channel is a vector, the activations of each channel are optimized
%  independently.
%
%  The output image, I, is a sequence of grayscale or truecolor (RGB)
%  images stored in a 4-D array. Images are concatenated along the fourth
%  dimension of I such that the image that maximizes the output of
%  channel(k) is I(:,:,:,k). 
%  
%  I = deepDreamImage(..., Name, Value) specifies additional name-value
%  pair arguments described below.
%
%  'InitialImage'   The image used to initialize Deep Dream. Use this to
%                   see how an image is modified to maximize network layer
%                   activations. The size of the inital image depends on
%                   the selected layer. For layers towards the end of the
%                   network, the image must be the same size or larger than
%                   the network's input size.
%
%                   Default: If you do not specify an image, the function
%                            uses a random initial image with pixel values
%                            drawn from a normal distribution with mean 0
%                            and standard deviation 1.
%
%  'PyramidLevels'  Number of multiresolution image pyramid levels to use
%                   to generate the output image. Increase the number of
%                   pyramid levels to produce larger output images at the
%                   expense of additional computation. Set the number of
%                   levels to 1 to produce an image the same size as
%                   'InitialImage'.
%
%                   Default: 3
%
%  'PyramidScale'   The scale between each pyramid level. Reduce the
%                   pyramid scale while increasing PyramidLevels to
%                   incorporate more fine-grain details into the
%                   synthesized image. This can help generate more
%                   informative images for layers at the beginning of the
%                   network.
%
%                   Default: 1.4
%
%  'NumIterations'  The number of iterations per pyramid level. Increase
%                   the number of iterations to produce more detailed
%                   images at the expense of additional computation.
% 
%                   Default: 10
%
%  'OutputScaling'  The type of scaling to apply to the output image. Valid
%                   values are 'linear' or 'none'. Select 'linear' to scale
%                   output pixel values between [0 1]. The output image
%                   corresponding to each layer channel, I(:,:,:,channel),
%                   is scaled independently. Select 'none' to disable
%                   output scaling.
%
%                   Default: 'linear'
%
%  'Verbose'        Set to true to display progress information.
%
%                   Default: true
%
%  'ExecutionEnvironment'  The execution environment for the network.
%                          Specify what hardware resources to use to run
%                          the optimization. Possible values are:
%
%                          'auto' - Use a GPU if available, otherwise use
%                                   the CPU.
%                          'gpu'  - Use the GPU. To use a GPU, you must
%                                   have Parallel Computing Toolbox(TM),
%                                   and a CUDA-enabled NVIDIA GPU with
%                                   compute capability 3.0 or higher. If a
%                                   suitable GPU is not available, the
%                                   function returns an error.
%                          'cpu'  - Use the CPU.
%
%                           Default: 'auto'
%
% Notes:
% ------
% - This function implements a version of Deep Dream that uses a
%   multi-resolution image pyramid and Laplacian Pyramid Gradient
%   Normalization to generate high-resolution images.
%
% - Selecting ReLU or dropout layers for visualization may not produce
%   useful images because of the effect those layers have on the network
%   gradients. 
%
% - To visualize classification layer features, select the last fully
%   connected layer before the classification layer.
%
% Example: Visualize convolutional neural network features
% ----------------------------------------------------------
% % Train a convolutional neural network on digit data. 
% [XTrain, TTrain] = digitTrain4DArrayData;
% 
% layers = [ ...
%     imageInputLayer([28 28 1])
%     convolution2dLayer(5,20)
%     reluLayer()
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(10)
%     softmaxLayer()
%     classificationLayer()];
%
% options = trainingOptions('sgdm', 'Plots', 'training-progress');
% net = trainNetwork(XTrain, TTrain, layers, options);
% net.Layers
%
% % Select the last fully-connected layer for visualization. Choosing this
% % layer illustrates what images the network thinks look like digits.
% layer = 'fc'
%
% % Select all ten output channels for visualization. 
% channels = 1:10;
%
% % Generate images.
% I = deepDreamImage(net, layer, channels);
%
% % Display the image corresponding to digit 0.
% figure
% imshow(I(:,:,:,1))
%
% See also trainNetwork, SeriesNetwork, alexnet, vgg16, vgg19.
 
% References
% ----------
% DeepDreaming with TensorFlow :
%    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb

% Copyright 2016-2017 The MathWorks, Inc.

[layerIdx, params] = iParseInputs(net, layer, channels, varargin{:});

% Get internal network.
iNet = nnet.internal.cnn.SeriesNetwork(...
    nnet.cnn.layer.Layer.getInternalLayers(net.Layers));

% Modify network to support image optimization.
iVisualNet = ...
    nnet.internal.cnn.visualize.VisualNetwork.createVisualNetworkForChannelAverage(...
    iNet, layerIdx, channels);

% Convert learnable parameters to the correct format
GPUShouldBeUsed = nnet.internal.cnn.util.GPUShouldBeUsed( ...
    params.ExecutionEnvironment );

if GPUShouldBeUsed
    executionSettings = struct( ...
    	'executionEnvironment', 'gpu', ...
    	'useParallel', false );
else
    executionSettings = struct( ...
    	'executionEnvironment', 'cpu', ...
    	'useParallel', false );
end

iVisualNet = iVisualNet.prepareNetworkForTraining( executionSettings );
    
numChannels = numel(iVisualNet.Layers{end}.Channels);

% Move data to the GPU.
if GPUShouldBeUsed
    X = gpuArray(single(repmat(params.InitialImage, [1,1,1,numChannels])));
else
    X = single(repmat(params.InitialImage, [1,1,1,numChannels]));
end
    
% Zero center initial image using average image stored in the network.
X = iSubtractAverageImage(iVisualNet, X);

X = nnet.internal.cnn.visualize.deepDreamImageLaplacianNorm(...
    iVisualNet, X, ...
    params.NumIterations, ...
    params.PyramidLevels, ...
    params.PyramidScale, ...
    params.TileSize,... 
    params.StepSize,...
    params.LaplacianGradNorm, ...
    params.Verbose);

% Add back the average image.
I = iAddAverageImage(iVisualNet, X);

% Scale output for display.
if ~strcmpi(params.OutputScaling, 'none')
    I = iScaleImage(I);
end

% Return data on the host.
I = gather(I);
end

%--------------------------------------------------------------------------
function [layerIdx, params] = iParseInputs(net, layer, channels, varargin)

% iCheckNetwork(net);

layerIdx = iCheckLayerAndReturnValidLayerIndex(net, layer);

iCheckChannels(net.Layers, layerIdx, channels);

p = inputParser;
addParameter(p, 'InitialImage',    []);
addParameter(p, 'NumIterations',   10);
addParameter(p, 'PyramidLevels',   3);
addParameter(p, 'PyramidScale',    1.4);
addParameter(p, 'Verbose',         true);
addParameter(p, 'OutputScaling',   'linear');
addParameter(p, 'ExecutionEnvironment', 'auto');

parse(p, varargin{:});

userInput = p.Results;

params.InitialImage = iValidateAndReturnInitialImage( p, net, layerIdx);

iCheckNumIterations(userInput.NumIterations);

iCheckPyramidLevels(userInput.PyramidLevels);

iCheckPyramidScale(userInput.PyramidScale);

iCheckLogicalParameter(userInput.Verbose, 'Verbose');

outputScaling = iCheckOutputScaling(userInput.OutputScaling);

params.ExecutionEnvironment = iValidateAndReturnExecutionEnvironment( ...
    userInput.ExecutionEnvironment);

% User visible parameters
params.NumIterations = double(userInput.NumIterations);
params.PyramidLevels = double(userInput.PyramidLevels);
params.PyramidScale  = double(userInput.PyramidScale);
params.Verbose       = logical(userInput.Verbose);
params.OutputScaling = char(outputScaling);

% Internal parameters
params.TileSize          = max(net.Layers(1).InputSize(1),net.Layers(1).InputSize(2));
params.StepSize          = 1;
params.LaplacianGradNorm = true;

end

%--------------------------------------------------------------------------
function iCheckNetwork(net)

validateattributes(net, {'SeriesNetwork'}, {'scalar'}, mfilename, 'network', 1);

assertNoSequenceInputLayer( net );

end

%--------------------------------------------------------------------------
function scaling = iCheckOutputScaling(value)
scaling = validatestring(value, {'none', 'linear'}, ...
    mfilename, 'OutputScaling');
end

%--------------------------------------------------------------------------
function layerIndex = iCheckLayerAndReturnValidLayerIndex(net, layerNameOrIndex)

internalLayers = nnet.cnn.layer.Layer.getInternalLayers(net.Layers);

layerIndex = iValidateNetworkLayerNameOrIndex(...
    layerNameOrIndex, internalLayers, mfilename);

end

%--------------------------------------------------------------------------
function iCheckChannels(layers, layerIndex, channels)
% net.Layers(layerIndex) must have specified channels

layerOutputSize = iLayerOuputSize(layers, layerIndex);

numChannels = layerOutputSize(3);

validateattributes(channels, {'numeric'}, ...
    {'vector', 'nonempty', 'real', 'nonsparse', 'integer', '>=', 1, '<=' numChannels}, ...
    mfilename, 'channel', 3);
end

%--------------------------------------------------------------------------
function iCheckInitialImage(I, net, layerIdx)

netSize = net.Layers(1).InputSize;

validateattributes(I, {'numeric'}, ...
    {'nonempty', 'real', 'nonsparse', 'size', [NaN NaN netSize(3)]}, mfilename, 'InitialImage');

iCheckInitialImageSize(net, layerIdx, size(I));
end

%--------------------------------------------------------------------------
function iCheckNumIterations(n)
validateattributes(n, {'numeric'}, ...
    {'scalar', 'integer', 'nonempty', 'real', 'positive', 'nonsparse'}, ...
    mfilename, 'NumIterations');
end

%--------------------------------------------------------------------------
function iCheckPyramidLevels(n)
validateattributes(n, {'numeric'}, ...
    {'scalar', 'integer', 'nonempty', 'real', 'positive', 'nonsparse'}, ...
    mfilename, 'PyramidLevels');
end

%--------------------------------------------------------------------------
function iCheckPyramidScale(n)
validateattributes(n, {'numeric'}, ...
    {'scalar', 'nonempty', 'real', '>', 1, 'finite', 'nonsparse'}, ...
    mfilename, 'PyramidScale');
end

%--------------------------------------------------------------------------
function iCheckLogicalParameter(tf,paramName)
validateattributes(tf, {'logical','numeric'},...
    {'nonnan', 'scalar', 'nonempty', 'real','nonsparse'},...
    mfilename,paramName);

end

%--------------------------------------------------------------------------
function iCheckInitialImageSize(net, layer, forwardSize)

minimumSize = nnet.internal.cnn.visualize.computeReceptiveFieldSize(...
        net, layer);

if(any(forwardSize(1:2) < minimumSize(1:2)))
    error(message('nnet_cnn:deepDreamImage:InitialImageNotValidImage', mat2str(minimumSize)));
end

end

%--------------------------------------------------------------------------
function initialImage = iValidateAndReturnInitialImage(p, net, layer)

% If no initial image is provided, then generate one
if ismember('InitialImage', p.UsingDefaults )
    
    receptiveFieldSize = ...
        nnet.internal.cnn.visualize.computeReceptiveFieldSize(...
        net, layer);
    
    initialImage = single(randn(receptiveFieldSize));            
else       
    
    % Else, subtract the mean value of the initial image, to give it
    % mean zero.
    % Load in the initial image, if given
    initialImage = p.Results.InitialImage;
    
    iCheckInitialImage(initialImage, net, layer);
    
    % Cast to precision used by optimization process.
    initialImage = single(initialImage);
    
end

end

%--------------------------------------------------------------------------
function validatedCharArray = iValidateAndReturnExecutionEnvironment(originalCharArray)
validExecutionEnvironments = {'auto', 'gpu', 'cpu'};
validatedCharArray = validatestring( ...
    originalCharArray, validExecutionEnvironments, ...
    'deepDreamImage', 'ExecutionEnvironment');
end

%--------------------------------------------------------------------------
% Subtract network average image from input. Resize average image to
% appropriate size if needed.
%--------------------------------------------------------------------------
function X = iSubtractAverageImage(internalNet, X)
if iHasAverageImage(internalNet)        
    sz = size(X);
    avgI = nnet.internal.cnn.visualize.resizeImage(...
        internalNet.Layers{1}.AverageImage, sz(1:2));  
    X = X - avgI;
end
end

%--------------------------------------------------------------------------
% Add network average image to input. Resize average image to
% appropriate size if needed.
%--------------------------------------------------------------------------
function X = iAddAverageImage(internalNet, X)
if iHasAverageImage(internalNet)
    sz = size(X);
    avgI = nnet.internal.cnn.visualize.resizeImage(...
        internalNet.Layers{1}.AverageImage, sz(1:2));    
    X = X + avgI;
end
end

%--------------------------------------------------------------------------
function tf = iHasAverageImage(internalNet)
tf = ~isempty( internalNet.Layers{1}.AverageImage );
end

%--------------------------------------------------------------------------
% Scale image such that the max value maps to 1 and min value maps to zero,
% similar to MAT2GRAY. Note that each layer channel is scaled independently.
%--------------------------------------------------------------------------
function B = iScaleImage(I)

numChannels = size(I,4);
B = zeros(size(I), 'like', I);
for i = 1:numChannels
    A = I(:,:,:,i);
    range = [min(A(:)) max(A(:))];
    delta = 1 ./ (range(2) - range(1));    
    B(:,:,:,i) = delta * I(:,:,:,i) - range(1) * delta;    
end
end

%--------------------------------------------------------------------------
function outputSize = iLayerOuputSize(layers, layerIdx)

internalLayers = nnet.cnn.layer.Layer.getInternalLayers(layers);

outputSize = layers(1).InputSize;

for i = 1:layerIdx
    outputSize = internalLayers{i}.forwardPropagateSize(outputSize);
end

end

%--------------------------------------------------------------------------
function layerIndex = iValidateNetworkLayerNameOrIndex(...
    layerNameOrIndex, internalLayers, fname)

if ischar(layerNameOrIndex)
    name = layerNameOrIndex;
    
    [layerIndex, layerNames] = nnet.internal.cnn.layer.Layer.findLayerByName(internalLayers, name);
    
    % first and last layer are not valid.
    layerNames = layerNames(2:end-1);
    
    try
        % pretty print error message. will print available layer names in
        % case of a mismatch.
        validatestring(name, layerNames, fname, 'layer');
    catch Ex
        throwAsCaller(Ex);
    end
    
    % Only 1 match allowed. This is guaranteed during construction of SeriesNetwork.
    assert(numel(layerIndex) == 1);
        
else
    % first and last layer index is not allowed.
    validateattributes(layerNameOrIndex, {'numeric'},...
        {'positive', 'integer', 'real', 'scalar', '>' 1, '<', numel(internalLayers)}, ...
        fname, 'layer');
    layerIndex = layerNameOrIndex;
end
end

%--------------------------------------------------------------------------
function assertNoSequenceInputLayer( net )
internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers( net.Layers );
isRNN = nnet.internal.cnn.util.isRNN( internalLayers );
if isRNN
    error(message('nnet_cnn:deepDreamImage:NotAvailableForRNN'));
end
end
