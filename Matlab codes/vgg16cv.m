function [t,x] =vgg16cv(path,optimizer,augmentation,numfold)
%resnet18cv Summary of this function goes here
%path :path folder
%optimizer:'adam','rmsprop','sgdm'
%augmentaion: 0 means there is no augmentation,and 1 means there is aug
%numfold: can take values as 2,3,4,5,...
%   example:
%[t,x]=resnet18cv('G:\new researches\dataset224','adam',0,5);

digitDatasetPath = fullfile(path);
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% Determine the split up
total_split=countEachLabel(imds)
% Number of Images
num_images=length(imds.Labels);

% Visualize random images
perm=randperm(num_images,6);
% for idx=1:length(perm)
%     
%     subplot(2,3,idx);
%     imshow(imread(imds.Files{perm(idx)}));
%     title(sprintf('%s',imds.Labels(perm(idx))))
%     
% end
%%K-fold Validation
% Number of folds
num_folds=numfold;

%for num=2:3
    
% Loop for each fold
for fold_idx=1:num_folds
    
    fprintf('Processing %d among %d folds \n',fold_idx,num_folds);
    
   % Test Indices for current fold
    test_idx=fold_idx:num_folds:num_images;

    % Test cases for current fold
    imdsTest = subset(imds,test_idx);
   labeltest=countEachLabel(imdsTest);
    % Train indices for current fold
    train_idx=setdiff(1:length(imds.Files),test_idx);
    
    % Train cases for current fold
    imdsTrain = subset(imds,train_idx);
    labeltrain= countEachLabel(imdsTrain);
    % nasnetmobile Architecture 
      net=vgg19;
     % Replacing the last layers with new layers
    layersTransfer = net.Layers(1:end-3);
    clear net;
 % Number of categories
    numClasses = numel(categories(imdsTrain.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

    
    % Preprocessing Technique
   
    % Training Options, we choose a small mini-batch size due to limited images 
    options = trainingOptions(optimizer,...
        'MaxEpochs',20,'MiniBatchSize',8,...
        'Shuffle','every-epoch', ...
        'InitialLearnRate',1e-4, ...
        'Verbose',false,...
        'Plots','training-progress');
    %'LearnRateSchedule','piecewise'
    %'OutputFcn',@(info)savetrainingplot(info)
    if augmentation==1
%     % Data Augumentation
      augmenter = imageDataAugmenter( ...
         'RandRotation',[-5 5],'RandXReflection',1,...
         'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
    
    % Resizing all training images to [224 224] for ResNet architecture
     auimds = augmentedImageDatastore([224 224],imdsTrain,'DataAugmentation',augmenter);
    
    % Training
    netTransfer = trainNetwork(auimds,layers,options);
    else
    netTransfer = trainNetwork(imdsTrain,layers,options);
    end
    % Resizing all testing images to [224 224] for ResNet architecture   
    % augtestimds = augmentedImageDatastore([224 224],imdsTest);
   
    % Testing and their corresponding Labels and Posterior for each Case
    [predicted_labels(test_idx),posterior(test_idx,:)] = classify(netTransfer,imdsTest);
    
    % Save the Independent ResNet Architectures obtained for each Fold
   % save(sprintf('googleNet_%d_among_%d_folds',fold_idx,num_folds),'netTransfer','test_idx','train_idx','labeltest','labeltrain');
    delete(findall(0))
    % Clearing unnecessary variables 
    clearvars -except optimizer path augmentation total_split numfold fold_idx num_folds num_images predicted_labels posterior imds netTransfer;
    
end
%analyzeNetwork(netTransfer)
%%Performance Study
% Actual Labels
actual_labels=imds.Labels;

% Confusion Matrix
%figure;
%plotconfusion(actual_labels,predicted_labels')
%title('Confusion Matrix');
%ROC CURVE
test_labels=double(nominal(imds.Labels));

% ROC Curve - Our target class is the first class in this scenario 
[fp_rate,tp_rate,T,AUC]=perfcurve(test_labels,posterior(:,1),1);
%figure;
%plot(fp_rate,tp_rate,'b-');
%grid on;
%xlabel('False Positive Rate');
%ylabel('Detection Rate');
% Area under the ROC curve value
AUC
%evaluation
%Evaluate(YValidation,YPred)
ACTUAL=actual_labels;
PREDICTED=predicted_labels';
idx = (ACTUAL()==total_split.Label(1));
%disp(idx)
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;
tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;

tp_rate = tp/p;
tn_rate = tn/n;

accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
t=[AUC,accuracy,sensitivity,specificity,precision,recall,f_measure,gmean];
x={t,path,optimizer,augmentation,numfold};
end

