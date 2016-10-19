addpath('./../');%!!!!!!!
addpath('./../../');%!!!!!!!

global DATAopts;
DATAopts = UCF101Init;
[allVids, labs, splits] = GetVideosPlusLabels('Challenge');

alpha_deepF=0.5;
alpha_handF=0.1;
datasetName='UCF101'

bazeDir='/home/ionut/asustor_ionut/Data/results/cvpr2017/forFusion_state_of_the_art/ucf101/videoRep/'



name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF101Layerconv5bMediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netC3DpcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
C3D_sp32cl256pca0=sp32cl256pca0;
sp32cl256pca0(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca0(:, end-(32*256) + 1 : end), alpha_deepF);
n_C3D_sp32cl256pca0=NormalizeRowsUnit(sp32cl256pca0);

name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF101Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netSpVGG19pcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
SpVGG19_sp32cl256pca0=sp32cl256pca0;
sp32cl256pca0(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca0(:, end-(32*256) + 1 : end), alpha_deepF);
n_SpVGG19_sp32cl256pca0=NormalizeRowsUnit(sp32cl256pca0);

name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF101Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netTempSplit1VGG16pcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
TempSplit1VGG16_sp32cl256pca0=sp32cl256pca0;
sp32cl256pca0(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca0(:, end-(32*256) + 1 : end), alpha_deepF);
n_TempSplit1VGG16_sp32cl256pca0=NormalizeRowsUnit(sp32cl256pca0);

name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF101Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netTempSplit2VGG16pcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
TempSplit2VGG16_sp32cl256pca0=sp32cl256pca0;
sp32cl256pca0(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca0(:, end-(32*256) + 1 : end), alpha_deepF);
n_TempSplit2VGG16_sp32cl256pca0=NormalizeRowsUnit(sp32cl256pca0);


name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF101Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netTempSplit3VGG16pcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
TempSplit3VGG16_sp32cl256pca0=sp32cl256pca0;
sp32cl256pca0(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca0(:, end-(32*256) + 1 : end), alpha_deepF);
n_TempSplit3VGG16_sp32cl256pca0=NormalizeRowsUnit(sp32cl256pca0);

clear sp32cl256pca0

name=[bazeDir 'FEVidHmgDenseBlockSize8_8_6_DatasetUCF101FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72SPfisherVector.mat']
load(name);
hmg_fv=fisherVector;
n_hmg_fv=NormalizeRowsUnit(PowerNormalization(fisherVector,alpha_handF ));
clear fisherVector

name=[bazeDir 'FEVid_IDTDatasetUCF101IDTfeatureHOFMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim54fisherVector.mat']
load(name);
hof_fv=fisherVector;
n_hof_fv=NormalizeRowsUnit(PowerNormalization(fisherVector,alpha_handF ));
clear fisherVector

name=[bazeDir 'FEVid_IDTDatasetUCF101IDTfeatureHOGMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat']
load(name);
hog_fv=fisherVector;
n_hog_fv=NormalizeRowsUnit(PowerNormalization(fisherVector,alpha_handF ));
clear fisherVector

name=[bazeDir 'FEVid_IDTDatasetUCF101IDTfeatureMBHxMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat']
load(name);
mbhx_fv=fisherVector;
n_mbhx_fv=NormalizeRowsUnit(PowerNormalization(fisherVector,alpha_handF ));
clear fisherVector

name=[bazeDir 'FEVid_IDTDatasetUCF101IDTfeatureMBHyMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat']
load(name);
mbhy_fv=fisherVector;
n_mbhy_fv=NormalizeRowsUnit(PowerNormalization(fisherVector,alpha_handF ));
clear fisherVector





%n_C3D_sp32cl256pca0 n_SpVGG19_sp32cl256pca0 n_TempSplit1VGG16_sp32cl256pca0 n_TempSplit2VGG16_sp32cl256pca0 n_TempSplit3VGG16_sp32cl256pca0
%n_hmg_fv
%n_hof_fv n_hog_fv n_mbhx_fv n_mbhy_fv

nEncoding=6;
allDist=cell(nEncoding, 1);


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0 ));
allDist{1,1}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit2VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0 ));
allDist{1,2}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit3VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0 ));
allDist{1,3}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv));
allDist{2,1}=temp * temp'; allDist{2,2}=allDist{2,1};  allDist{2,3}=allDist{2,1};


temp=NormalizeRowsUnit( cat(2, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv, n_hmg_fv));
allDist{3,1}=temp * temp'; allDist{3,2}=allDist{3,1};  allDist{3,3}=allDist{3,1};


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hmg_fv ));
allDist{4,1}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit2VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hmg_fv  ));
allDist{4,2}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit3VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hmg_fv  ));
allDist{4,3}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv ));
allDist{5,1}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit2VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv));
allDist{5,2}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit3VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv ));
allDist{5,3}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv, n_hmg_fv ));
allDist{6,1}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit2VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv, n_hmg_fv));
allDist{6,2}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit3VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv, n_hmg_fv ));
allDist{6,3}=temp * temp';


clear temp

all_clfsOut=cell(nEncoding,3);
all_accuracy=cell(nEncoding,3);

mean_all_accuracy=cell(nEncoding,1);

cRange = 100;
nReps = 1;
nFolds = 3;



parpool(3);
%%
for k=1:nEncoding
    k
    parfor i=1:3
        
        trainI = splits(:,i) == 1;
        
       if ~isempty(strfind(datasetName, 'HMDB51'))
            testI  = splits(:,i) == 2;
       elseif ~isempty(strfind(datasetName, 'UCF101'))
            testI=~trainI;
       end
       
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        trainDist = allDist{k,i}(trainI, trainI);
        testDist = allDist{k,i}(testI, trainI);
        

        [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
        %fprintf('accuracy: %.3f\n', accuracy);
    end
     all_clfsOut{k}=clfsOut;
     all_accuracy{k}=accuracy;
     fprintf('Accuracy for encoding %d: %.3f\n',k, mean((all_accuracy{k}{1} + all_accuracy{k}{2} + all_accuracy{k}{3})./3));
end

delete(gcp('nocreate')) %///
%%%%

clear allDist


finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('%.3f\n', finalAcc(j));

    
end

