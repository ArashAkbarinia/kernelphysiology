function [net, info] = CnnChess(varargin)
% CnnChess   
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
opts.nclass = 3;
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

if isempty(opts.expDir)
  opts.expDir = fullfile(opts.masterDir, '/nets/chess/');
end
if isempty(opts.dataDir)
  opts.dataDir = fullfile(opts.masterDir, '/datasets/chess/');
end
if isempty(opts.imdb)
  opts.imdb = fullfile(opts.dataDir, 'imdb.mat');
else
  opts.imdb = fullfile(opts.dataDir, opts.imdb);
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

opts.InputSize = [rows, cols, chns];

% preparing model and data
switch opts.modelType
  case 'lenet'
    net = ChessTrainingInit(opts);
  case 'nin'
    net = ChessTrainingInitNin('networkType', opts.networkType);
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
save(sprintf('%s/%s', opts.expDir, NetworkName), '-struct', 'net');

[CurrentPath, ~, ~] = fileparts(mfilename('fullpath'));
copyfile(sprintf('%s/ChessTrainingInit.m', CurrentPath), opts.expDir);

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
