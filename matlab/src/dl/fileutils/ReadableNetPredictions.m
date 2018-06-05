function ActivationReport = ReadableNetPredictions(ActivationReport, DatasetName)
%ReadableNetPredictions  Converts the ID of a class to a human readble name
%   Detailed explanation goes here

if ~strcmpi(DatasetName, 'ilsvrc-test')
  error('No labels are available for dataset: %s\n', DatasetName);
end

% reading the labels from vgg16
vggnet = vgg16;

nImages = size(ActivationReport, 2);

ContrastNames = fieldnames(ActivationReport(1).cls);
nContrasts = numel(ContrastNames);

ImagenetClasses = vggnet.Layers(41).ClassNames;

parfor i = 1:nImages
  for j = 1:nContrasts
    prediction = ActivationReport(i).cls.(ContrastNames{j}).prediction.type;
    % for vgg16-3c and vgg16-5c the first six characters are 'class'
    prediction = ImagenetClasses{str2double(prediction(6:end))};
    ActivationReport(i).cls.(ContrastNames{j}).prediction.type = prediction;
  end
end

end
