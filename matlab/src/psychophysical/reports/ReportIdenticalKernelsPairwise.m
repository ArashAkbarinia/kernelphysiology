function [] = ReportIdenticalKernelsPairwise(FolderPath)
%REPORTIDENTICALKERNELSPAIRWISE Summary of this function goes here
%   Detailed explanation goes here

FilePattern = 'VariationAtLayers';
MatFiles = dir(sprintf('%s%s*.mat', FolderPath, FilePattern));

nfiles = numel(MatFiles);

for i = 1:nfiles
  VariationAtLayerXX = load(sprintf('%s%s', FolderPath, MatFiles(i).name));
  means = VariationAtLayerXX.stats.means;
  fprintf('%.2f %.2f %.2f %.2f %.2f\n', mean(means(~isnan(means(:, 1)), 1:5)));
end

end
