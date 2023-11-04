 %% Try to classify something else
 clear;clc
 %format bank
%load('Densenet_1_among_5_folds.mat')
%load('xception_1_among_5_folds.mat')%0.138454 seconds.170 layers
%load('ALEXNET_1_among_5_folds.mat')%0.045648  seconds.25 layers
%load('darknet19_1_among_5_folds.mat')%0.068242 seconds.65 layers
%load('googlenet_1_among_5_folds.mat')%0.018441 seconds. seconds.144 layers
%load('ResNet18_1_among_5_folds.mat')%0.007419 seconds 41 layers
%load('ResNet50_1_among_5_folds.mat')%0.021940 seconds.177 layers
%load('nasnetmobile_2_among_5_folds.mat')%0.245589  seconds 913 layers
%load('newcustomisedmodel_1_among_5_folds.mat')%0.004950 seconds.7 layer
%load('vgg16_1_among_5_folds.mat')%0.046677  seconds.44 layers
%img = readimage(imds,100);
load('darknet19_1_among_5_folds.mat')
[filerootd, pathname1, filterindex1] = uigetfile({'*.jpg';'*.png';'*.jpeg'}, ...
   'Select an image');
x=imresize(imread([pathname1, filerootd]),[256 256]);
% x=imresize(imread('G:\covid project\matlabdb80\bacteria\person1_bacteria_1.jpeg'),[224 224]);
%x=imresize(imread('G:\covid project\matlabdb80\covid\01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg'),[224 224]);
[a,b,c]=size(x);
if c==1
  img=cat(3,x,x,x);
else
    img=x;
end
%actualLabel = imds.Labels(100);
actualLabel='--';
%predictedLabel = trainedNet.classify(img);
%YPred=classify(trainedNet,img);
tic;[YPred,scores] = classify(netTransfer,img,'ExecutionEnvironment','GPU');toc
%[YPred,scores] =netTransfer.classify(img);
%YPred = predict(trainedNet,img);
switch(YPred)
    case 'COVID'
       score=scores(1); 
    case 'NON-COVID'
        score=scores(2);
     
end
imshow(img);
title(['Predicted: ' char(YPred) mat2str(floor(score*100)) '%',' Actual: ' char(actualLabel)])



