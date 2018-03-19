function [ ] = ContrastCorrelationFolder(FolderPath)
%ContrastCorrelationFolder Summary of this function goes here
%   Detailed explanation goes here

FileList = dir([FolderPath, filesep, '*.mat']);

nfiles = numel(FileList);
for i = 1:nfiles
  CurrentNet = load([FolderPath, filesep, FileList(i).name]);
  
  CorrMat = ContrastCorrelation(CurrentNet);
  [nKernels, KernelSize] = GetKernelInfo(CurrentNet);
  
  PosCont = mean(mean(abs(CorrMat)));
  fprintf('[%s]\n', FileList(i).name);
  fprintf('\tAverage contrast coreelations R=%.3f G=%.3f B=%.3f. All=%.3f.\n', PosCont, mean(PosCont));
  fprintf('\t#kernels=%i kernel size=%ix%i.\n', nKernels, KernelSize);
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
