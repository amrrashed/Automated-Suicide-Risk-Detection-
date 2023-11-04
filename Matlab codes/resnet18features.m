clc;close all;clear
%load images
%crop224new
digitDatasetPath = fullfile('G:\new researches\mansour paper\crop224new');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% Determine the split up
total_split=countEachLabel(imds)
% Number of Images
num_images=length(imds.Labels);
[imdsTrain,imdsTest] = splitEachLabel(imds,0.8,'randomized');
 numClasses = numel(categories(imdsTrain.Labels));
%    % Data Augumentation
%     augmenter = imageDataAugmenter( ...
%         'RandRotation',[-5 5],'RandXReflection',1,...
%         'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
%     
%    % Resizing all training images to [224 224] for ResNet architecture
    %imds = augmentedImageDatastore([227 227],imds);
%  
    net =resnet18;
    %net = inceptionresnetv2;%86%
    %analyzeNetwork(net)
    %nasnetmoblile 79%
    layer = 'pool5';
featuresTrain = activations(net,imdsTrain,layer,'OutputAs','rows');%,'ExecutionEnvironment','cpu'
featuresTest = activations(net,imdsTest,layer,'OutputAs','rows');
featuresall=[featuresTrain;featuresTest];
whos featuresTrain
 YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
Yall = [YTrain;YTest];
%cat to numeric values
keys = categorical({'NORMAL', 'SUICIDE'});
values = [0, 1];
[found, where] = ismember(Yall, keys)
v = nan(size(Yall));
v(found) = values(where(found));
%merge all features in one file
all=[featuresall,v];
csvwrite('resnet18featurescrop.csv',all,1)
%Fit Image Classifier
%Use the features extracted from the training images as predictor variables and fit a multiclass support vector machine (SVM) using fitcecoc (Statistics and Machine Learning Toolbox).
classifier = fitcecoc(featuresTrain,YTrain);
%Classify Test Images
%Classify the test images using the trained SVM model using the features extracted from the test images.

YPred = predict(classifier,featuresTest);
%Display four sample test images with their predicted labels.

idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end
    accuracy = mean(YPred == YTest)