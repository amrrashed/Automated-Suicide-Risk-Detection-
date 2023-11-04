 clear;clc
 %format bank
load('Densenet_1_among_5_folds.mat')
%netTransfer.Layers
%analyzeNetwork(netTransfer)
%img = readimage(imds,100);
[filerootd, pathname1, filterindex1] = uigetfile({'*.bmp';'*.jpg';'*.png';'*.jpeg'}, ...
   'Select an image');
inputSize = netTransfer.Layers(1).InputSize(1:2);
img=imresize(imread([pathname1, filerootd]),inputSize);
figure
imshow(img)
act1 = activations(netTransfer,img,'conv2_block1_1_conv');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 8]);
figure
imshow(I)