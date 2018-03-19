function report = ContrastCorrelationSubfolders(FolderPath)
%ContrastCorrelationFolder Summary of this function goes here
%   Detailed explanation goes here

subfolders = GetSubFolders(FolderPath);

nfiles = numel(subfolders);

report = zeros(nfiles, 8);
for i = 1:nfiles
  CurrentNet = load([FolderPath, filesep, subfolders{i}, filesep, 'netfinal.mat']);
  CurrentInf = load([FolderPath, filesep, subfolders{i}, filesep, 'infofinal.mat']);
  
  CorrMat = ContrastCorrelation(CurrentNet);
  [nKernels, KernelSize] = GetKernelInfo(CurrentNet);
  
  PosCont = mean(mean(abs(CorrMat)));
  
  report(i, 1:3) = PosCont;
  report(i, 4) = mean(PosCont);
  report(i, 5:6) = 1 - [CurrentInf.val(end).top1err, CurrentInf.val(end).top5err];
  report(i, 7) = numel(CurrentInf.val);
  report(i, 8) = CurrentNet.meta.trainOpts.learningRate(1);
  fprintf('[%s]\n', subfolders{i});
  fprintf('\tAverage contrast coreelations R=%.3f G=%.3f B=%.3f. All=%.3f.\n', PosCont, report(i, 4));
  fprintf('\tAccuracy=%.3f Accuracy(5)=%.3f\n', report(i, 5:6));
  fprintf('\t#kernels=%i kernel size=%ix%i #epochs=%d.\n', nKernels, KernelSize, numel(CurrentInf.val));
end

end

function [nKernels, KernelSize] = GetKernelInfo(net)

% TODO: implement proper layers
layers = 1;

% NOTICE: deciding according to the field 'params' might not be the best.
if isfield(net, 'params')
  [KernelSize(1), KernelSize(2), ~, nKernels] = size(net.params(layers).value);
else
  [KernelSize(1), KernelSize(2), ~, nKernels] = size(net.layers{layers}.weights{1, 1});
end

end
