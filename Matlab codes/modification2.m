clc;clear
digitDatasetPath = fullfile('G:\new researches\mansour paper\dataset256');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
labels=countEachLabel(imds);
img2=zeros(256,256,3);
parfor i=1:length(imds.Labels)
img=readimage(imds,i);
img1 = uint8(imresize(img,[256 256]));
c=length(size(img));
if c==2
img2=cat(3,img1,img1,img1);
imwrite(img2,cell2mat(imds.Files(i)))
elseif c==3
 imwrite(img1,cell2mat(imds.Files(i)))
end
%solve bitdepth error
%aa=imfinfo(imds.Files{perms(i)});
%B(i)=aa.BitDepth;
end

