clc;clear all
% n is the number of subjects
n = 3;
% You can press stop button manually on tranining plot(on top right corner besides number of iterations) once accuracy reaches upto desired level
% looping through all subjects and cropping faces if found
% extract the subject photo and crop faces and saving it in to respective
% folders
for i =1:n
    str = ['s0',int2str(i)];
    ds1 = imageDatastore('E:\CIT project\CODES\photos\','IncludeSubfolders',true,'LabelSource','foldernames');
    cropandsave(ds1,str);
end
%  im = imageDatastore('E:\CIT project\CODES\croppedfaces\','IncludeSubfolders',true,'LabelSource','foldernames');
%  % Resize the images to the input size of the net
%  im.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
%  [Train ,Test] = splitEachLabel(im,0.8,'randomized');
%  fc = fullyConnectedLayer(n);
%  net = alexnet;
%  ly = net.Layers;
%  ly(23) = fc;
%  cl = classificationLayer;
%  ly(25) = cl; 
%  % options for training the net if your newnet performance is low decrease
%  % the learning_rate
%  learning_rate = 0.00001;
%  opts = trainingOptions("rmsprop","InitialLearnRate",learning_rate,'MaxEpochs',5,'MiniBatchSize',64,'Plots','training-progress');
%  [newnet,info] = trainNetwork(Train, ly, opts);
%  [predict,scores] = classify(newnet,Test);
%  names = Test.Labels;
%  pred = (predict==names);
%  s = size(pred);
%  acc = sum(pred)/s(1);
%  fprintf('The accuracy of the test set is %f %% \n',acc*100);
% % Test a new Image
% % use code below with giving path to your new image
% % img = imread('...\img.jpg');
% % [img,face] = cropface(img);
% % face value is 1 when it detects face in image or 0
% % if face == 1
% %   img = imresize(img,[227 227]);
% %   predict = classify(newnet,img)
% % end
% % nameofs01 = 'name of subject 1';
% % nameofs02 = 'name of subject 2';
% % nameofs03 = 'name of subject 3';
% % if predict=='s01'
% %   fprintf('The face detected is %s',nameofs01);
% % elseif  predict=='s02'
% %   fprintf('The face detected is %s',nameofs02);
% % elseif  predict=='s03'
% %   fprintf('The face detected is %s',nameofs03);
% % end	 
% %  can use [predict,score] = classify(newnet,img) here score says the percentage that how confidence it is
