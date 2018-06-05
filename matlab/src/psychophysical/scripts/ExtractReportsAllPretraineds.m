% extracting and printing accuracy and each contrast level

NetNames = {'vgg16', 'vgg19', 'alexnet', 'googlenet', 'inceptionv3', 'resnet50', 'resnet101'};

DatasetName = 'ilsvrc-test';
AnalysisDir = '/mnt/sdb/Dropbox/Gießen/KarlArash/dnn/analysis/kernelsactivity/';

NumNets = numel(NetNames);

% in the first experiment we tried it with the first five convolutional
% layers
AccuracuReports = zeros(5, NumNets);

for i = 1:NumNets
  NetwrokName = NetNames{i};
  outdir = [AnalysisDir, NetwrokName, '/'];
   
  if exist(sprintf('%s/%s/AccuracyReport.mat', outdir, DatasetName), 'file')
    MatPath = sprintf('%s/%s/AccuracyReport.mat', outdir, DatasetName);
    AccuracyReport = load(MatPath);
  else
    MatPath = sprintf('%s/%s/ActivationReport.mat', outdir, DatasetName);
    ActivationReport = load(MatPath, 'ActivationReport');
    AccuracyReport = ReportAccuracyAtContrastLevel(ActivationReport.ActivationReport, DatasetName);
    
    save(sprintf('%s/%s/AccuracyReport.mat', outdir, DatasetName), '-struct', 'AccuracyReport');
  end
  
  FormatedString = [repmat('%.2f ', [1, size(AccuracyReport.ContrastAccuracyMatrix, 2)]), '\n'];
  fprintf(FormatedString, permute(mean(AccuracyReport.ContrastAccuracyMatrix, 1), [2, 1]));
  AccuracuReports(:, i) = permute(mean(AccuracyReport.ContrastAccuracyMatrix, 1), [2, 1]);
end
