function AllKernels = DeepDreamContrastCorrelation(FolderPath)
%DeepDreamContrastCorrelation Summary of this function goes here
%   Detailed explanation goes here

ImageList = dir(sprintf('%s*.png', FolderPath));

kers = 1000;
ImageList = ImageList(1:kers);

imi = imread(sprintf('%s%s', FolderPath, ImageList(1).name));
[rows, cols, chns] = size(imi);
AllKernels = zeros(rows, cols, chns, kers, 'uint8');

for i = 1:kers
  imi = imread(sprintf('%s%s', FolderPath, ImageList(i).name));
  
  AllKernels(:, :, :, i) = imi;
end

save(sprintf('%sAllKernels.mat', FolderPath), 'AllKernels');

end
