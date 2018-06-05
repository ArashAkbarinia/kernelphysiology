% extracting and printing the pairwise comparison of the maximum level of
% contrast to all other levels

NetNames = {'inceptionv3', 'vgg19', 'vgg16', 'vgg3c4x', 'resnet50', 'resnet101', 'googlenet', 'alexnet'};

DatasetName = 'ilsvrc-test';
AnalysisDir = '/mnt/sdb/Dropbox/Gießen/KarlArash/dnn/analysis/kernelsactivity/';

for j = 1:numel(NetNames)
  NetwrokName = NetNames{j};
  outdir = [AnalysisDir, NetwrokName, '/'];
  
  MatPath = sprintf('%s/%s/ActivationReport.mat', outdir, DatasetName);
  ActivationReport = load(MatPath, 'ActivationReport');
  ActivationReport = ActivationReport.ActivationReport;
  
  for t = 1:10
    if exist(sprintf('%s/%s/VariationAtLayers%.2d.mat', outdir, DatasetName, t), 'file')
    else
      parfor i = 1:50000
        AverageKernelMatchingsEqTop(i) = ContrastVsAccuracy(ActivationReport(i), false, [1:t - 1, t + 1:10]);
      end
      
      meds = zeros(50000, 5);
      means = zeros(50000, 5);
      stds = zeros(50000, 5);
      for i = 1:50000
        means(i, :) = AverageKernelMatchingsEqTop(i).avg;
      end
      VariationAtLayers.AverageKernelMatchingsEqTop = AverageKernelMatchingsEqTop;
      VariationAtLayers.stats.meds = meds;
      VariationAtLayers.stats.means = means;
      VariationAtLayers.stats.stds = stds;
      
      save(sprintf('%s/%s/VariationAtLayers%.2d.mat', outdir, DatasetName, t), '-struct', 'VariationAtLayers');
    end
    
    fprintf('%.2f %.2f %.2f %.2f %.2f\n', mean(means(~isnan(means(:, 1)), 1:5)));
    fprintf('\t%.2f %.2f %.2f %.2f %.2f\n', std(means(~isnan(means(:, 1)), 1:5)));
  end
end