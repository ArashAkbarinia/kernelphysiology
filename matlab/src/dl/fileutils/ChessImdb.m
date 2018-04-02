function imdb = ChessImdb(FolderPath, ImdbPath, FileFormat)
%CHESSIMDB Summary of this function goes here
%   Detailed explanation goes here

AllSubFolders = GetSubFolders(FolderPath);
nSubFolders = numel(AllSubFolders);
AllImageList = cell(nSubFolders, 1);
for i = 1:nSubFolders
  AllImageList{i, 1} = dir(sprintf('%s%s/*.%s', FolderPath, AllSubFolders{i}, FileFormat));
end

[TrainList, TestList] = ChooseSetList(AllImageList);

if strcmpi(FileFormat, 'txt')
  [data, labels, set] = ReadTxts(TrainList, TestList);
else
  GreyScale = true;
  [data, labels, set] = ReadImages(TrainList, TestList, GreyScale);
end

data = single(data);
labels = single(labels);
set = single(set);

imdb.images.data = data;
imdb.images.labels = labels;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = AllSubFolders;

save(ImdbPath, '-struct', 'imdb', '-v7.3') ;

end

function [data, labels, set] = ReadTxts(TrainList, TestList)

ntrain = size(TrainList, 1);
ntest = size(TestList, 1);
ntotal = ntrain + ntest;

rows = 8;
cols = 8;

data = single(zeros(rows, cols, 1, ntotal));
labels = single(zeros(1, ntrain + ntest));
set = single(zeros(1, ntrain + ntest));

[data, labels, set] = ReadOneSetTxt(TrainList, data, labels, set, 0, 1);
[data, labels, set] = ReadOneSetTxt(TestList, data, labels, set, ntrain, 3);

end

function [data, labels, set] = ReadOneSetTxt(WhichList, data, labels, set, ntrain, WhichSet)

for i = 1:size(WhichList, 1)
  TmpName = WhichList{i, 1};
  
  cimg = dlmread(TmpName);
  
  j = i + ntrain;
  data(:, :, :, j) = single(cimg);
  labels(1, j) = single(WhichList{i, 2});
  set(1, j) = single(WhichSet);
end

end

function [data, labels, set] = ReadImages(TrainList, TestList, GreyScale)

if nargin < 3
  GreyScale = true;
end

ntrain = size(TrainList, 1);
ntest = size(TestList, 1);
ntotal = ntrain + ntest;

im0 = imread(TrainList{1, 1});
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

for i = 1:size(WhichList, 1)
  TmpName = WhichList{i, 1};
  
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
  labels(1, j) = single(WhichList{i, 2});
  set(1, j) = single(WhichSet);
end

end

function [TrainList, TestList] = ChooseSetList(AllImageList)

TrainList = {};
TestList = {};

t1 = 20000;
t2 = 10000;
t3 = 1000;
MaxTest = 5000;

for i = 1:numel(AllImageList)
  cl = AllImageList{i};
  NumImages = numel(cl);
  if NumImages > t1
    TrainMax = t1;
  elseif NumImages > t2
    TrainMax = t2;
  else
    TrainMax = t3;
  end
  
  for j = 1:TrainMax
    TrainList{end + 1, 1} = sprintf('%s/%s', cl(j).folder, cl(j).name);
    TrainList{end + 0, 2} = i;
  end
  for j = TrainMax + 1:min(NumImages, TrainMax + MaxTest)
    TestList{end + 1, 1} = sprintf('%s/%s', cl(j).folder, cl(j).name);
    TestList{end + 0, 2} = i;
  end
end

end
