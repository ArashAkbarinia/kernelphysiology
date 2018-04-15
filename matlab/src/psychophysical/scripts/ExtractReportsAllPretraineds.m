NetNames = {'vgg16', 'vgg19', 'alexnet', 'googlenet', 'inceptionv3', 'resnet50', 'resnet101'};

DatasetName = 'ilsvrc-test';
AnalysisDir = '/mnt/sdb/Dropbox/Gießen/KarlArash/dnn/analysis/kernelsactivity/';

for i = 1:numel(NetNames)
  NetwrokName = NetNames{i};
  outdir = [AnalysisDir, NetwrokName, '/'];
  
  MatPath = sprintf('%s/%s/ActivationReport.mat', outdir, DatasetName);
  ActivationReport = load(MatPath, 'ActivationReport');
  AccuracyReport = ReportAccuracyAtContrastLevel(ActivationReport.ActivationReport, DatasetName);
  
  save(sprintf('%s/%s/AccuracyReport.mat', outdir, DatasetName), '-struct', 'AccuracyReport');
  
  FormatedString = [repmat('%.2f ', [1, size(AccuracyReport.ContrastAccuracyMatrix, 2)]), '\n'];
  fprintf(FormatedString, permute(mean(AccuracyReport.ContrastAccuracyMatrix, 1), [2, 1]));
end
