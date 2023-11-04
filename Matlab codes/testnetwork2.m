 %% Try to classify something else
 clear;clc
 %format bank
 %load('D:\CIT project\CODES\ResNet18_1_among_5_folds.mat')
 %load('newcustomisedmodel_2_among_5_folds.mat')
 %load('ALEXNET_1_among_5_folds.mat')
 load('darknet19_1_among_5_folds.mat')
%img = readimage(imds,100);
[filerootd, pathname1, filterindex1] = uigetfile({'*.bmp';'*.jpg';'*.png';'*.jpeg'}, ...
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
%actualLabel = imds.Labels(100);
actualLabel='--';
%predictedLabel = trainedNet.classify(img);
%YPred=classify(trainedNet,img);
%[YPred,scores] = classify(trainedNet,img)
%[YPred,score] =trainedNet.classify(img);
%YPred = predict(trainedNet,img);
h = figure;
h.Position(3) = 2*h.Position(3);
ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);
ax2.ActivePositionProperty = 'position';

image(ax1,img)
[label,score] = classify(netTransfer,img);
title(ax1,{char(label),num2str(max(score),2)});
%Select the top five predictions by selecting the classes with the highest scores.

[~,idx] = sort(score,'descend');
idx = idx(2:-1:1);
classes = netTransfer.Layers(end).Classes;
classNamesTop = string(classes(idx));
scoreTop = score(idx);
%%%


%Display the top five predictions as a histogram.

barh(ax2,scoreTop)
xlim(ax2,[0 1])
title(ax2,'Top 2')
xlabel(ax2,'Probability')
yticklabels(ax2,classNamesTop)
ax2.YAxisLocation = 'right';

%Use occlusionSensitivity to determine which parts of the image positively influence the classification result.

scoreMap = occlusionSensitivity(netTransfer,img,label,'ExecutionEnvironment','cpu' );

%Plot the result over the original image with transparency to see which areas of the image affect the classification score.

figure
imshow(img,'InitialMagnification', 150)
hold on
imagesc(scoreMap,'AlphaData',0.5);
colormap jet
colorbar



