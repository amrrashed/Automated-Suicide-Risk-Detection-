 clear;clc
 %format bank
%load('vgg16_1_among_5_folds.mat')
%load('Densenet_1_among_5_folds.mat')
load('ALEXNET_1_among_5_folds.mat')
%netTransfer.Layers
%analyzeNetwork(netTransfer)
%img = readimage(imds,100);
[filerootd, pathname1, filterindex1] = uigetfile({'*.bmp';'*.jpg';'*.png';'*.jpeg'}, ...
   'Select an image');

inputSize = netTransfer.Layers(1).InputSize(1:2);
img=imresize(imread([pathname1, filerootd]),inputSize);

[classfn,score] = classify(netTransfer,img);
figure
imshow(img);
title(sprintf("%s (%.2f)", classfn, score(classfn)));
% lgraph = layerGraph(netTransfer);
% lgraph = removeLayers(lgraph, lgraph.Layers(end).Name);
 layersTransfer = netTransfer.Layers(1:end-1);
 lgraph = layerGraph(layersTransfer);
 dlnet = dlnetwork(lgraph);
softmaxName = 'softmax';
featureLayerName = 'conv5';%conv5_block32_1_relu
dlImg = dlarray(single(img),'SSC');
[featureMap, dScoresdMap] = dlfeval(@gradcam, dlnet, dlImg, softmaxName, featureLayerName, classfn);
gradcamMap = sum(featureMap .* sum(dScoresdMap, [1 2]), 3);
gradcamMap = extractdata(gradcamMap);
gradcamMap = rescale(gradcamMap);
gradcamMap = imresize(gradcamMap, inputSize, 'Method', 'bicubic');
figure
imshow(img);
hold on;
imagesc(gradcamMap,'AlphaData',0.5);
colormap jet
hold off;
title("Grad-CAM");

