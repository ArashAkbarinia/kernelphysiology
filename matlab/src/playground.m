%%
anet = alexnet;
vnet = vgg16;

inim2 = imread('/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2012/Val_Images_RGB/ILSVRC2012_val_00000002.png');
inim7 = imread('/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2012/Val_Images_RGB/ILSVRC2012_val_00000007.png');

outdir = [];

%% playing with images of 2 and 7 of ILSVRC dataset
areport2 = ActivationCorrDifferentContrasts(anet, inim2, outdir, false);
areport7 = ActivationCorrDifferentContrasts(anet, inim7, outdir, false);
vreport2 = ActivationCorrDifferentContrasts(vnet, inim2, outdir, false);
vreport7 = ActivationCorrDifferentContrasts(vnet, inim7, outdir, false);

%% plotting the results of VGG16 and AlexNet
metric = 'KernelCorrAvg';
a(1:2, :) = permute(vreport2.metrices.(metric)(1:2, 3, :), [1,3,2]);
a(4:5, :) = permute(vreport7.metrices.(metric)(1:2, 3, :), [1,3,2]);
a(7:8, 1:15) = permute(areport2.metrices.(metric)(1:2, 3, :), [1,3,2]);
a(10:11, 1:15) = permute(areport7.metrices.(metric)(1:2, 3, :), [1,3,2]);
figure, plot(a([1,2,4,5,], :)'), legend
figure, plot(a([7,8,10,11,], 1:15)'), legend

%% converting old reports

for i = 1:50000
  ActivationReport.data{i, 1}.cls = ActivationReport2(i).cls;
  
  ActivationReport.data{i, 1}.metrices.PixelTopAvg = ActivationReport2(i).CompMatrix;
  ActivationReport.data{i, 1}.metrices.PixelHistAvg = ActivationReport2(i).CompMatrixHist;
end

ActivationReport.info.nImages = 50000;
ActivationReport.info.nContrasts = 11;
ActivationReport.info.nLayers	= size(ActivationReport.data{1, 1}.metrices.PixelTopAvg, 3);

%%
save(['ActivationReport.mat'], 'ActivationReport');
