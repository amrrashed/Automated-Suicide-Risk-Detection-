clc;clear ;close all
path='D:\new researches\SUICIDE PAPER\Dataset\our DB1\crop224new';
digitDatasetPath = fullfile(path);
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
 [XTrain,XValidation] = splitEachLabel(imds,0.8);
 YTrain=XTrain.Labels;
 YValidation=XValidation.Labels;
% Determine the split up
total_split=countEachLabel(imds)
% Number of Images
num_images=length(imds.Labels);
optimVars = [
    optimizableVariable('SectionDepth',[1 3],'Type','integer')
    optimizableVariable('InitialLearnRate',[1e-2 1],'Transform','log')
    optimizableVariable('Momentum',[0.8 0.98])
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];
ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation);
BayesObject = bayesopt(ObjFcn,optimVars, ...
    'MaxTime',14*60*60, ...
    'IsObjectiveDeterministic',false, ...
    'UseParallel',false);
%Load the best network found in the optimization and its validation accuracy.
bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
valError = savedStruct.valError
%%%%%%%%%%%
[YPredicted,probs] = classify(savedStruct.trainedNet,XTest);
testError = 1 - mean(YPredicted == YTest)
%%%%%%%%%
NTest = numel(YTest);
testErrorSE = sqrt(testError*(1-testError)/NTest);
testError95CI = [testError - 1.96*testErrorSE, testError + 1.96*testErrorSE]
%%%%%%%%%%%%
figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(YTest,YPredicted);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';