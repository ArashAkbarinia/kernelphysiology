function FigureHandler = PlotEpochAccuracy(FolderPath, FigureName)
%PlotEpochAccuracy  Plotting the training and validation accuracy.
%   Detailed explanation goes here

if nargin < 2
  FigureName = 'Accuracy Curves';
else
  FigureName = [FigureName, ' Accuracy Curves'];
end

SubFolders = GetSubFolders(FolderPath);

nFiles = numel(SubFolders);

FigRows = round(sqrt(nFiles));
FigCols = ceil(sqrt(nFiles));

FigureHandler(1) = figure('NumberTitle', 'off', 'name', FigureName);
FigureHandler(2) = figure('NumberTitle', 'off', 'name', [FigureName, ' Compact']);

colours = DistinguishableColours(nFiles);

for i = 1:nFiles
  FileName = sprintf('%s/%s/log.csv', FolderPath, SubFolders{i});
  CurrentLog = readtable(FileName, 'Delimiter', ';');
  
  figure(FigureHandler(1));
  subplot(FigRows, FigCols, i);
  hold on;
  plot(CurrentLog.epoch, CurrentLog.acc);
  plot(CurrentLog.epoch, CurrentLog.val_acc);
  title(SubFolders{i}, 'Interpreter', 'none');
  
  figure(FigureHandler(2));
  subplot(1, 2, 1);
  hold on;
  plot(CurrentLog.epoch, CurrentLog.acc, 'color', colours(i, :));

  subplot(1, 2, 2);
  hold on;
  plot(CurrentLog.epoch, CurrentLog.val_acc, 'color', colours(i, :));
end

end
