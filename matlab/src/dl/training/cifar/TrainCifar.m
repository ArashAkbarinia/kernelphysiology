function [CifarNet, CifarInfo] = TrainCifar(imdb, CheckPointPath, ResumeTraining)
%TrainCifar Summary of this function goes here
%   Detailed explanation goes here
% https://es.mathworks.com/help/vision/examples/object-detection-using-deep-learning.html

if nargin < 3
  ResumeTraining = false;
end

if ~exist(CheckPointPath, 'dir')
  mkdir(CheckPointPath);
end

images = imdb.images;
meta = imdb.meta;

TrainingImages = images.data(:, :, :, images.set == 1);
TrainingImages = uint8(TrainingImages);
TestImages = images.data(:, :, :, images.set == 3);
TestImages = uint8(TestImages);

TrainingLabels = categorical(meta.classes(images.labels(images.set == 1)));
TestLabels = categorical(meta.classes(images.labels(images.set == 3)));

if ResumeTraining
  AllCheckPoints = dir(sprintf('%sconvnet_checkpoint*', CheckPointPath));
  LastCheckPoint = load(sprintf('%s%s', CheckPointPath, AllCheckPoints(end).name));
  layers = LastCheckPoint.net.Layers;
else
  NumImageCategories = numel(meta.classes);
  
  % Create the image input layer for 32x32x3 CIFAR-10 images
  [rows, cols, chns, ~] = size(TrainingImages);
  
  ImageSize = [rows, cols, chns];
  InputLayer = imageInputLayer(ImageSize);
  
  % Convolutional layer parameters
  FilterSize = [5, 5];
  NumFilters = 32;
  
  MiddleLayers = ConstructMiddleLayers(FilterSize, NumFilters);
  
  NumFcls = 64;
  FinalLayers = ConstructFinalLayers(NumFcls, NumImageCategories);
  
  layers = [InputLayer; MiddleLayers; FinalLayers];
  
  % Initialize the first convolutional layer weights using normally
  % distributed random numbers with standard deviation of 0.0001.
  % This helps improve the convergence of training.
  layers(2).Weights = 0.0001 * randn([FilterSize, chns, NumFilters]);
end

% Set the network training options
opts = trainingOptions ...
  ( ...
  'sgdm', ...
  'Momentum', 0.9, ...
  'InitialLearnRate', 0.001, ...
  'LearnRateSchedule', 'piecewise', ...
  'LearnRateDropFactor', 0.1, ...
  'LearnRateDropPeriod', 8, ...
  'L2Regularization', 0.004, ...
  'MaxEpochs', 40, ...
  'MiniBatchSize', 128, ...
  'Verbose', true, ...
  'Plots', 'training-progress', ...
  'ValidationData', {TestImages, TestLabels},...
  'ValidationFrequency', 50, ...
  'ValidationPatience', Inf, ...
  'CheckpointPath', CheckPointPath ...
  );

% train a network.
[CifarNet, CifarInfo] = trainNetwork(TrainingImages, TrainingLabels, layers, opts);

save(sprintf('%s/CifarNet.mat', CheckPointPath), 'CifarNet');
save(sprintf('%s/CifarInfo.mat', CheckPointPath), 'CifarInfo');

end

function MiddleLayers = ConstructMiddleLayers(FilterSize, NumFilters)

MiddleLayers = ...
  [
  % The first convolutional layer has a bank of 32 5x5x3 filters. A
  % symmetric padding of 2 pixels is added to ensure that image borders
  % are included in the processing. This is important to avoid
  % information at the borders being washed away too early in the
  % network.
  convolution2dLayer(FilterSize, NumFilters, 'Padding', 2);
  
  % Note that the third dimension of the filter can be omitted because it
  % is automatically deduced based on the connectivity of the network. In
  % this case because this layer follows the image layer, the third
  % dimension must be 3 to match the number of channels in the input
  % image.
  
  % Next add the ReLU layer:
  reluLayer();
  
  % Follow it with a max pooling layer that has a 3x3 spatial pooling area
  % and a stride of 2 pixels. This down-samples the data dimensions from
  % 32x32 to 15x15.
  maxPooling2dLayer(3, 'Stride', 2);
  
  % Repeat the 3 core layers to complete the middle of the network.
  convolution2dLayer(FilterSize, NumFilters, 'Padding', 2);
  reluLayer();
  maxPooling2dLayer(3, 'Stride', 2);
  
  convolution2dLayer(FilterSize, 2 * NumFilters, 'Padding', 2);
  reluLayer();
  maxPooling2dLayer(3, 'Stride', 2);
  ];

end

function FinalLayers = ConstructFinalLayers(NumFcls, NumImageCategories)

FinalLayers = ...
  [
  % Add a fully connected layer with 64 output neurons. The output size of
  % this layer will be an array with a length of 64.
  fullyConnectedLayer(NumFcls);
  
  % Add an ReLU non-linearity.
  reluLayer();
  
  % Add the last fully connected layer. At this point, the network must
  % produce 10 signals that can be used to measure whether the input image
  % belongs to one category or another. This measurement is made using the
  % subsequent loss layers.
  fullyConnectedLayer(NumImageCategories);
  
  % Add the softmax loss layer and classification layer. The final layers use
  % the output of the fully connected layer to compute the categorical
  % probability distribution over the image classes. During the training
  % process, all the network weights are tuned to minimize the loss over this
  % categorical distribution.
  softmaxLayer();
  classificationLayer();
  ];

end
