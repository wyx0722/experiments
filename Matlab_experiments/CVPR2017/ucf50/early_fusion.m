


% global DATAopts;
% DATAopts = UCFInit;
% 
% 
% [allVids, labs, groups] = GetVideosPlusLabels('Full');
% 



alpha_deepF=0.5;
alpha_handF=0.1;


bazeDir='/home/ionut/asustor_ionut/Data/results/cvpr2017/ucf50/videoRep/'



name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF50Layerconv5bMediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netC3DpcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
%C3D_sp32cl256pca0=sp32cl256pca0;
sp32cl256pca0(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca0(:, end-(32*256) + 1 : end), alpha_deepF);
n_C3D_sp32cl256pca0=NormalizeRowsUnit(sp32cl256pca0);

name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF50Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netSpVGG19pcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
%SpVGG19_sp32cl256pca0=sp32cl256pca0;
sp32cl256pca0(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca0(:, end-(32*256) + 1 : end), alpha_deepF);
n_SpVGG19_sp32cl256pca0=NormalizeRowsUnit(sp32cl256pca0);

name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF50Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256nettempVGG16pcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
%TempSplit1VGG16_sp32cl256pca0=sp32cl256pca0;
sp32cl256pca0(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca0(:, end-(32*256) + 1 : end), alpha_deepF);
n_TempSplit1VGG16_sp32cl256pca0=NormalizeRowsUnit(sp32cl256pca0);

clear sp32cl256pca0

name=[bazeDir 'FEVidHmgDenseBlockSize8_8_6_DatasetUCF50FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72__fisherAll.mat']
load(name);
%hmg_fv=fisherAll;
n_hmg_fv=NormalizeRowsUnit(PowerNormalization(fisherAll,alpha_handF ));
clear fisherAll

name=[bazeDir 'FEVid_IDTDatasetUCF50IDTfeatureHOFMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim54__fisherAll.mat']
load(name);
%hof_fv=fisherAll;
n_hof_fv=NormalizeRowsUnit(PowerNormalization(fisherAll,alpha_handF ));
clear fisherAll

name=[bazeDir 'FEVid_IDTDatasetUCF50IDTfeatureHOGMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__fisherAll.mat']
load(name);
%hog_fv=fisherAll;
n_hog_fv=NormalizeRowsUnit(PowerNormalization(fisherAll,alpha_handF ));
clear fisherAll

name=[bazeDir 'FEVid_IDTDatasetUCF50IDTfeatureMBHxMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__fisherAll.mat']
load(name);
%mbhx_fv=fisherAll;
n_mbhx_fv=NormalizeRowsUnit(PowerNormalization(fisherAll,alpha_handF ));
clear fisherAll

name=[bazeDir 'FEVid_IDTDatasetUCF50IDTfeatureMBHyMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__fisherAll.mat']
load(name);
%mbhy_fv=fisherAll;
n_mbhy_fv=NormalizeRowsUnit(PowerNormalization(fisherAll,alpha_handF ));
clear fisherAll


nPar=5;
alpha=0.1;
n_representation=NormalizeRowsUnit(PowerNormalization(representation, alpha));

nEncoding=8;
allDist=cell(1, nEncoding);


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0 ));
allDist{1}=temp * temp';



temp=NormalizeRowsUnit( cat(2, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv));
allDist{2}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv, n_hmg_fv));
allDist{3}=temp * temp'; 


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hmg_fv ));
allDist{4}=temp * temp';



temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv ));
allDist{5}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv, n_hmg_fv ));
allDist{6}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_representation ));
allDist{7}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hmg_fv, n_representation ));
allDist{8}=temp * temp';




all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;

parpool(nPar);
for k=1:nEncoding
k
% 
% Leave-one-group-out cross-validation
parfor i=1:max(groups)
    testI = groups == i;
    trainI = ~testI;
    trainDist = allDist{k}(trainI, trainI);
    testDist = allDist{k}(testI, trainI);
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

all_clfsOut{k}=clfsOut;
all_accuracy{k}=accuracy;

fprintf('Accuracy for encoding %d: %.3f\n',k, mean(mean(cat(2, accuracy{:}))'));

end

delete(gcp('nocreate'))

clear allDist



finalAcc=zeros(1,nEncoding);
for j=1:nEncoding

    finalAcc(j)=mean(mean(cat(2, all_accuracy{j}{:}), 2));
    fprintf('%.3f\n', finalAcc(j));

    
end

 
 
 
