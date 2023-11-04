clc;clear all;close all
%load('Darknet19_1_among_5_folds.mat')%%CT images
load('nasnetmobile_1_among_5_folds.mat')%%xray images
[filerootd, pathname1, filterindex1] = uigetfile({'*.bmp';'*.jpg';'*.png';'*.jpeg'}, ...
   'Select an image');
inputSize = netTransfer.Layers(1).InputSize(1:2);
img=imresize(imread([pathname1, filerootd]),inputSize);

[label,score] = classify(netTransfer,img);
figure
imshow(img);
title(sprintf("%s (%.2f)", label, score(label)));
scoreMap = imageLIME(netTransfer,img,label);
figure
imshow(img)
hold on
imagesc(scoreMap,'AlphaData',0.5)
colormap jet

