function imdb = ChessImdb(TrainFile, TestFile)
%CHESSIMDB Summary of this function goes here
%   Detailed explanation goes here

opts.ContrastNormalisation = false;
opts.WhitenData = false;
opts.ImdbPath = '/home/arash/Software/repositories/chesscnn/matlab/data/datasets/imdb128-org-mean.mat';

TrainList = ReadSetList(TrainFile);
TestList = ReadSetList(TestFile);

[data, labels, set] = ReadImages(TrainList, TestList);

data = single(data);
labels = single(labels);
set = single(set);

imdb.images.data = data;
imdb.images.labels = labels;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = {'white', 'black', 'draw'};

save(opts.ImdbPath, '-struct', 'imdb', '-v7.3') ;

end

function [data, labels, set] = ReadImages(TrainList, TestList, GreyScale)

if nargin < 3
  GreyScale = true;
end

ntrain = numel(TrainList{1});
ntest = numel(TestList{1});
ntotal = ntrain + ntest;

im0 = imread(TrainList{1, 1}{1});
[rows, cols, chns] = size(im0);
if GreyScale
  chns = 1;
end
npixels = rows * cols;

data = single(zeros(rows, cols, chns, ntotal));
labels = single(zeros(1, ntrain + ntest));
set = single(zeros(1, ntrain + ntest));

[data, labels, set] = ReadOneSet(TrainList, data, labels, set, 0, npixels, GreyScale, 1);
[data, labels, set] = ReadOneSet(TestList, data, labels, set, ntrain, npixels, GreyScale, 3);

end

function [data, labels, set] = ReadOneSet(WhichList, data, labels, set, ntrain, npixels, GreyScale, WhichSet)

for i = 1:numel(WhichList{1})
  TmpName = WhichList{1, 1}{i};
  
  cimg = imread(TmpName);
  
  if size(cimg, 3) == 1
    disp(TmpName);
    continue;
  end
  
  if GreyScale
    cimg = rgb2gray(cimg);
  end
  
  cimg = StandardImage(cimg, npixels);
  
  j = i + ntrain;
  data(:, :, :, j) = single(cimg);
  labels(1, j) = single(WhichList{1, 2}(i));
  set(1, j) = single(WhichSet);
end

end

function SetList = ReadSetList(FilePath)

fid = fopen(FilePath);
SetList = textscan(fid, '%s%d', 'Delimiter', ' ');
fclose(fid);

end
