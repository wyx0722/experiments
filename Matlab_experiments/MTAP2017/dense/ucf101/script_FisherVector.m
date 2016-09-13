

basePath='/home/ionut/asustor_ionut/'
addpath('./../');%!!!!!!!

typeFeature=cell(1,1);
% typeFeature{1}=@FEVidHmgDense
% typeFeature{2}=@FEVidHofDense
% typeFeature{3}=@FEVidHogDense
% typeFeature{4}=@FEVidMBHxDense
typeFeature{1}=@FEVidMBHyDense


fsr=[1 2 3 6];

all_accuracy=cell(length(typeFeature),length(fsr));
all_clfsOut=cell(length(typeFeature),length(fsr));


mType='Vid';

normStrategy='ROOTSIFT';
datasetName='UCF101';%!!!!!!!change
cl=256;
nPar=5;
alpha=0.1;%!!!!!!!change
d=72;
savePath=[basePath 'Data/results/mtap2017/ucf101/']%!!!!!!!change
%fsr=?;





for i=1:length(typeFeature)
    i
    for j=1:length(fsr)
        j
         [all_accuracy{i}{j}, all_clfsOut{i}{j}] = FisherVectorFramework_hmdb51_UCF101(typeFeature{i},mType, normStrategy,datasetName,cl, nPar, alpha, savePath, fsr(j));
    end
end
