function OutImage = CropCentreSquareImage(InImage)
%CropCentreSquareImage Summary of this function goes here
%   Detailed explanation goes here

[rows, cols, ~] = size(InImage);

if rows > cols
  dp = rows - cols;
  
  sp = floor(dp / 2) + 1;
  ep = cols + sp - 1;
  
  OutImage = InImage(sp:ep, :, :);
elseif cols > rows
  dp = cols - rows;
  
  sp = floor(dp / 2) + 1;
  ep = rows + sp - 1;
  
  OutImage = InImage(:, sp:ep, :);
else
  OutImage = InImage;
end

end
