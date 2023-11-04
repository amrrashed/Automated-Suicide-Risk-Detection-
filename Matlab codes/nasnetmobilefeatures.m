clc; close all; clear
% Load images
% crop224new
digitDatasetPath = fullfile('D:\new researches\SUICIDE PAPER\Dataset\our DB1\crop224new');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Determine the split up
total_split = countEachLabel(imds);

% Number of Images
num_images = length(imds.Labels);

[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

numClasses = numel(categories(imdsTrain.Labels));

% Load nasnetmobile
net = nasnetmobile;

% layers = net.Layers;
% layerNames = {layers.Name};
% disp(layerNames);
%layerNames(910)

% Layer for feature extraction (you may need to adjust this)
layer = 'global_average_pooling2d_1';

featuresTrain = activations(net, imdsTrain, layer, 'OutputAs', 'rows');
featuresTest = activations(net, imdsTest, layer, 'OutputAs', 'rows');
featuresall = [featuresTrain; featuresTest];

whos featuresTrain

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
Yall = [YTrain; YTest];

% Map categorical labels to numeric values
keys = categorical({'NORMAL', 'SUICIDE'});
values = [0, 1];
[found, where] = ismember(Yall, keys);
v = nan(size(Yall));
v(found) = values(where(found));

% Merge all features in one file
all = [featuresall, v];
csvwrite('nasnetmobile_features_cropped.csv', all, 1);

% Fit Image Classifier
% Use the features extracted from the training images as predictor variables
% and fit a multiclass support vector machine (SVM).
classifier = fitcecoc(featuresTrain, YTrain);

% Classify Test Images
% Classify the test images using the trained SVM model using the features
% extracted from the test images.
YPred = predict(classifier, featuresTest);

% Display sample test images with their predicted labels
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2, 2, i)
    I = readimage(imdsTest, idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

% Calculate accuracy
accuracy = mean(YPred == YTest);
