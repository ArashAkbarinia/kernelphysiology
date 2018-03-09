function FigureHandler = PlotAllKernels( net, layer )
%PlotAllKernels Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
  layer = 1;
end
if nargin < 3
  isvisible = 'on';
end

[rows, cols, chns, kers] = size(net.layers{layer, 1}.weights{1, 1});

FigRows = round(sqrt(kers));
FigCols = ceil(sqrt(kers));

FigureHandler = zeros(chns);
for c = 1: chns
  FigureHandler(c) = figure('name', ['All kernels channel ', num2str(c)], 'visible', isvisible);
  for i = 1:kers
    subplot(FigRows, FigCols, i);
    keri = net.layers{1, 1}.weights{1, 1}(:, :, c, i);
    
%     keri = NormaliseChannel(keri);    
%     keri(keri < 0) = -1;
%     keri(keri > 0) = +1;
    
    imshow(keri, []);
    title(['Kernel = ', num2str(i)]);
  end
end

end
