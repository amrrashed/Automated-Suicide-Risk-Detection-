clc;clear
digitDatasetPath = fullfile('G:\new researches\mansour paper\dataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
labels=countEachLabel(imds);
parfor i=1:length(imds.Labels)
img=readimage(imds,i);
[img2,face] = cropface(img);
imwrite(img2,cell2mat(imds.Files(i)))
end

