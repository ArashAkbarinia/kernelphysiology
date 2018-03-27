net = vgg16;

% path of the dataset
DatasetPath = '/home/arash/Software/repositories/kernelphysiology/data/computervision/voc2012/JPEGImages/';

ImageList = dir(sprintf('%s*.jpg', DatasetPath));

NumImages = numel(ImageList);

outdir = '/home/arash/Software/repositories/kernelphysiology/analysis/kernelsactivity/vgg16/voc2012/';

%%
NumAnalysedImages = 2;

% SelectedImages = randi(NumImages, [1, NumAnalysedImages]);
SelectedImages = [18:22, 24:26, 29, 31];

for i = SelectedImages
  inim = imread([DatasetPath, ImageList(i).name]);
  ImageOutDir = sprintf('%s%s/', outdir, ImageList(i).name(1:end - 4));
  ActivationReport = ActivationDifferentContrasts(net, inim, ImageOutDir);
end

%%

for i = SelectedImages
  inim = imread([DatasetPath, ImageList(i).name]);
  ImageOutDir = sprintf('%s%s/', outdir, ImageList(i).name(1:end - 4));
  ActivationReport = load([ImageOutDir, 'ActivationReport.mat']);
  AverageKernelMatching = ContrastVsAccuracy(ActivationReport);
end
