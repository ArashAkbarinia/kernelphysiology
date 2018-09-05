run /home/arash/Software/repositories/kernelphysiology/matlab/src/kernelphysiology.m

NetNames = {'vgg16', 'vgg19', 'alexnet', 'googlenet', 'inceptionv3', 'resnet50', 'resnet101', 'vgg3c4x'};

DatasetName = 'ilsvrc-test';
AnalysisDir = '/home/deeplearning/Desktop/maxresults/';
% AnalysisDir = '/home/arash/Software/repositories/kernelphysiology/analysis/kernelsactivity/';
% DatasetName = 'ilsvrc2017';

for i = 1:numel(NetNames)
  NetwrokName = NetNames{i};
  outdir = [AnalysisDir, NetwrokName, '/'];
  mkdir(outdir);
  fprintf('Processing network %s\n', NetwrokName);
  DatasetActivationDifferentContrasts(NetwrokName, DatasetName, outdir);
end
