% addpath('./../');%!!!!!!!
% addpath('./../../');%!!!!!!!
% 
% global DATAopts;
% DATAopts = HMDB51Init;
% [allVids, labs, splits] = GetVideosPlusLabels();
% 
% 
% alpha=0.5;
% datasetName='HMDB51'





alpha_deepF=0.5;
alpha_handF=0.1;


bazeDir='/home/ionut/asustor_ionut/Data/results/cvpr2017/hmdb51/clusters/videoRep/'



name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerconv5bMediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetC3DpcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl256.mat']
load(name);
%C3D_sp32cl256=sp32cl256;
sp32cl256(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256(:, end-(32*256) + 1 : end), alpha_deepF);
n_C3D_sp32cl256=NormalizeRowsUnit(sp32cl256);

name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetSpVGG19pcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl256.mat']
load(name);
%SpVGG19_sp32cl256=sp32cl256;
sp32cl256(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256(:, end-(32*256) + 1 : end), alpha_deepF);
n_SpVGG19_sp32cl256=NormalizeRowsUnit(sp32cl256);

name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetTempSplit1VGG16pcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl256.mat']
load(name);
%TempSplit1VGG16_sp32cl256=sp32cl256;
sp32cl256(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256(:, end-(32*256) + 1 : end), alpha_deepF);
n_TempSplit1VGG16_sp32cl256=NormalizeRowsUnit(sp32cl256);

clear sp32cl256


bazeDir='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/dense/'

name=[bazeDir 'FEVidHmgDenseBlockSize8_8_6_DatasetHMDB51FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72SPfisherVector.mat']
load(name);
%hmg_fv=fisherVector;
n_hmg_fv=NormalizeRowsUnit(PowerNormalization(fisherVector,alpha_handF ));
clear fisherVector

bazeDir='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/iDT/'

name=[bazeDir 'FEVid_IDTDatasetHMDB51IDTfeatureHOFMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim54fisherVector.mat']
load(name);
%hof_fv=fisherVector;
n_hof_fv=NormalizeRowsUnit(PowerNormalization(fisherVector,alpha_handF ));
clear fisherVector

name=[bazeDir 'FEVid_IDTDatasetHMDB51IDTfeatureHOGMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat']
load(name);
%hog_fv=fisherVector;
n_hog_fv=NormalizeRowsUnit(PowerNormalization(fisherVector,alpha_handF ));
clear fisherVector

name=[bazeDir 'FEVid_IDTDatasetHMDB51IDTfeatureMBHxMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat']
load(name);
%mbhx_fv=fisherVector;
n_mbhx_fv=NormalizeRowsUnit(PowerNormalization(fisherVector,alpha_handF ));
clear fisherVector

name=[bazeDir 'FEVid_IDTDatasetHMDB51IDTfeatureMBHyMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat']
load(name);
%mbhy_fv=fisherVector;
n_mbhy_fv=NormalizeRowsUnit(PowerNormalization(fisherVector,alpha_handF ));
clear fisherVector


nPar=5;
alpha=0.1;
n_representation=NormalizeRowsUnit(PowerNormalization(representation, alpha));

nEncoding=8;
allDist=cell(1, nEncoding);


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256, n_SpVGG19_sp32cl256, n_C3D_sp32cl256 ));
allDist{1}=temp * temp';



temp=NormalizeRowsUnit( cat(2, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv));
allDist{2}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv, n_hmg_fv));
allDist{3}=temp * temp'; 


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256, n_SpVGG19_sp32cl256, n_C3D_sp32cl256, n_hmg_fv ));
allDist{4}=temp * temp';



temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256, n_SpVGG19_sp32cl256, n_C3D_sp32cl256, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv ));
allDist{5}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256, n_SpVGG19_sp32cl256, n_C3D_sp32cl256, n_hof_fv, n_hog_fv, n_mbhx_fv, n_mbhy_fv, n_hmg_fv ));
allDist{6}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256, n_SpVGG19_sp32cl256, n_C3D_sp32cl256, n_representation ));
allDist{7}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256, n_SpVGG19_sp32cl256, n_C3D_sp32cl256, n_hmg_fv, n_representation ));
allDist{8}=temp * temp';

clear temp

%each row for the cell represents the results for all 3 splits
all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);
clfsOut=cell(1,nEncoding);
accuracy=cell(1,nEncoding);
%mean_all_clfsOut=cell(nEncoding,1);
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
        
        trainDist = allDist{k}(trainI, trainI);
        testDist = allDist{k}(testI, trainI);
        

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

