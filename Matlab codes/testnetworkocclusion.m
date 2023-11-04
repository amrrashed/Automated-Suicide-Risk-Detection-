 %% Try to classify something else
 clear;clc
 %format bank
 load('ResNet50_1_among_5_folds.mat')
% load('vgg16_1_among_5_folds.mat')
%opengl('save','software') 
%img = readimage(imds,100);
[filerootd, pathname1, filterindex1] = uigetfile({'*.jpeg';'*.bmp';'*.jpg';'*.png'}, ...
   'Select an image');
x=imread([pathname1, filerootd]);
% x=imresize(imread('G:\covid project\matlabdb80\bacteria\person1_bacteria_1.jpeg'),[224 224]);
%x=imresize(imread('G:\covid project\matlabdb80\covid\01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg'),[224 224]);

[a,b,c]=size(x);
if c==1
  img=cat(3,x,x,x);
else
    img=x;
end
classes = netTransfer.Layers(end).Classes;
[YPred,scores] = classify(netTransfer,img);
[~,topIdx] = maxk(scores, 2);
topScores = scores(topIdx);
topClasses = classes(topIdx);
figure(1)
imshow(img)
titleString = compose("%s (%.2f)",topClasses,topScores');
title(sprintf(join(titleString, "; ")));
map = occlusionSensitivity(netTransfer,img,YPred,'ExecutionEnvironment','cpu');
figure(2)
imshow(img,'InitialMagnification', 150)
hold on
imagesc(map,'AlphaData',0.5)
colormap jet
colorbar

title(sprintf("Occlusion sensitivity (%s)", ...
    YPred))

