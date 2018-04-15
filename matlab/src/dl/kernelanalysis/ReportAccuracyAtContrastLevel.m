function AccuracyReport = ReportAccuracyAtContrastLevel(ActivationReport, DatasetName)
%ReportAccuracyAtContrastLevel Summary of this function goes here
%   Detailed explanation goes here

if strcmpi(DatasetName, 'ilsvrc-test')
  ValidationDir = '/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2012/validation/';
  
  labels = dlmread(sprintf('%sILSVRC2012_validation_ground_truth.txt', ValidationDir));
  ImageInfos = load(sprintf('%sILSVRC2012_validation_meta.mat', ValidationDir));
else
  error('No groundtruth is available for dataset: %s\n', DatasetName);
end

nImages = size(ActivationReport, 2);

ContrastNames = fieldnames(ActivationReport(1).cls);
nContrasts = numel(ContrastNames);

AccuracyReport = struct();

ContrastAccuracyMatrix = zeros(nImages, nContrasts);
parfor i = 1:nImages
  AcceptedResults = strsplit(ImageInfos.synsets(labels(i)).words, ', ');
  for j = 1:nContrasts
    prediction = ActivationReport(i).cls.(ContrastNames{j}).prediction.type;
    MatchedAny = find(strcmpi(AcceptedResults, prediction), 1);
    ContrastAccuracyMatrix(i, j) = ~isempty(MatchedAny);
  end
end

AccuracyReport.ContrastAccuracyMatrix = ContrastAccuracyMatrix;
AccuracyReport.ContrastNames = ContrastNames;

end
