function [net, info] = CnnCifar(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

% model parameters
opts.modelType = 'lenet';
opts.networkType = 'simplenn';
% data organisation
opts.masterDir = '';
opts.expDir = '';
opts.dataDir = '';
opts.imdb = '';
opts.nclass = 10;
% cpu or gpu
opts.train.gpus = [];
% database creation
opts.whitenData = true;
opts.contrastNormalization = true;
% experiment
opts.continue = 0;
opts.BatchSize = 256;
opts.AugmentV1 = false;
opts.ExperimentName = 'test';

[opts, ~] = vl_argparse(opts, varargin);

if isempty(opts.dataDir)
  opts.dataDir = fullfile(opts.masterDir, sprintf('/datasets/cifar/cifar%d/', opts.nclass));
end
if isempty(opts.imdb)
  opts.imdb = fullfile(opts.dataDir, 'imdb.mat');
else
  opts.imdb = fullfile(opts.dataDir, opts.imdb);
end
if isempty(opts.expDir)
  opts.expDir = fullfile(opts.masterDir, sprintf('/nets/cifar/cifar%d/', opts.nclass));
end
NetworkName = sprintf('ex-%s-%s', opts.modelType, opts.networkType);
[~, ImdbName, ~] = fileparts(opts.imdb);
opts.expDir = fullfile(opts.expDir, sprintf('%s-%s-%s/', NetworkName, ImdbName, opts.ExperimentName));

% getting the imdb
if exist(opts.imdb, 'file')
  imdb = load(opts.imdb);
else
  error('Wrong IMDB path ''%s''.', opts.imdb);
end

[rows, cols, chns, nims] = size(imdb.images.data);

if opts.AugmentV1
  imdb = AugmentImages(imdb);
end

opts.InputSize = [rows, cols, chns];

% preparing model and data
switch opts.modelType
  case 'lenet'
    net = CifarTrainingInit(opts);
  case 'nin'
    net = cnn_cifar_init_nin('networkType', opts.networkType);
  otherwise
    error('Unknown model type ''%s''.', opts.modelType);
end

net.meta.classes.name = imdb.meta.classes(:)';

% training
switch opts.networkType
  case 'simplenn'
    trainfn = @CnnTrainSimple;
  case 'dagnn'
    trainfn = @CnnTrainDag;
end

[net, info] = trainfn(net, imdb, getBatch(opts), 'continue', opts.continue, 'expDir', opts.expDir, net.meta.trainOpts, opts.train, 'val', find(imdb.images.set == 3)); %#ok

net.layers{end}.type = 'softmax';
save(sprintf('%s/cifar%d-%s', opts.expDir, opts.nclass, NetworkName), '-struct', 'net');

[CurrentPath, ~, ~] = fileparts(mfilename('fullpath'));
copyfile(sprintf('%s/CifarTrainingInit.m', CurrentPath), opts.expDir);

end

function imdb = AugmentImages(imdb)

TrainingSetInds = imdb.images.set == 1;
TrainingSet = imdb.images.data(:, :, :, TrainingSetInds);
TrainingLabels = imdb.images.labels(TrainingSetInds);

imdbgpu = gpuArray(TrainingSet);

g00 = Gaussian2Gradient1(0.5, 0);
g90 = Gaussian2Gradient1(0.5, pi / 2);

im00 = gather(imfilter(imdbgpu, g00));
im90 = gather(imfilter(imdbgpu, g90));

%   dataMean = mean(im00(:, :, :, :), 4);
%   im00 = bsxfun(@minus, im00, dataMean);
%
%   dataMean = mean(im90(:, :, :, :), 4);
%   im90 = bsxfun(@minus, im90, dataMean);

imdb.images.data = cat(4, imdb.images.data, im00, im90);
imdb.images.labels = cat(2, imdb.images.labels, TrainingLabels, TrainingLabels);

% clearing
clear imdbgpu im00 im90 ;

si = nims + 1;
ei = size(imdb.images.labels, 2);
imdb.images.set(si:ei) = 1;

%   save('imdb-augmented.mat', '-struct', 'imdb');

end

function fn = getBatch(opts)

switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x, y) getSimpleNNBatch(x, y);
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus));
    fn = @(x, y) getDagNNBatch(bopts, x, y);
end

end

function [images, labels] = getSimpleNNBatch(imdb, batch)

images = imdb.images.data(:, :, :, batch);
labels = imdb.images.labels(1, batch);
if rand > 0.5
  images = fliplr(images);
end

end

function inputs = getDagNNBatch(opts, imdb, batch)

images = imdb.images.data(:, :, :, batch);
labels = imdb.images.labels(1, batch);
if rand > 0.5
  images = fliplr(images);
end
if opts.numGpus > 0
  images = gpuArray(images);
end
inputs = {'input', images, 'label', labels};

end
