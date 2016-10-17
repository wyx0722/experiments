addpath('./../');%!!!!!!!
addpath('./../../');%!!!!!!!

global DATAopts;
DATAopts = UCF101Init;
[allVids, labs, splits] = GetVideosPlusLabels('Challenge');

alpha=0.5;
datasetName='UCF101'

bazeDir='/home/ionut/asustor_ionut/Data/results/cvpr2017/ucf101/videoRep/'



name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF101Layerconv5bMediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netC3DpcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
C3D_sp32cl256pca0=sp32cl256pca0;

name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF101Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netSpVGG19pcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
SpVGG19_sp32cl256pca0=sp32cl256pca0;


name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF101Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netTempSplit1VGG16pcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
TempSplit1VGG16_sp32cl256pca0=sp32cl256pca0;

name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF101Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netTempSplit2VGG16pcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
TempSplit2VGG16_sp32cl256pca0=sp32cl256pca0;


name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF101Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netTempSplit3VGG16pcaDim256_0_spClusters32_64_sp_clDim_check32_64___sp32cl256pca0.mat']
load(name);
TempSplit3VGG16_sp32cl256pca0=sp32cl256pca0;


clear sp32cl256pca0




nEncoding=1;
allDist=cell(nEncoding, 1);


t1=TempSplit1VGG16_sp32cl256pca0;
t1(:, end-(32*256) + 1 :end)=PowerNormalization(t1(:, end-(32*256) + 1 : end), 0.5);

t2=TempSplit2VGG16_sp32cl256pca0;
t2(:, end-(32*256) + 1 :end)=PowerNormalization(t2(:, end-(32*256) + 1 : end), 0.5);

t3=TempSplit3VGG16_sp32cl256pca0;
t3(:, end-(32*256) + 1 :end)=PowerNormalization(t3(:, end-(32*256) + 1 : end), 0.5);

t4=SpVGG19_sp32cl256pca0;
t4(:, end-(32*256) + 1 :end)=PowerNormalization(t4(:, end-(32*256) + 1 : end), 0.5);

t5=C3D_sp32cl256pca0;
t5(:, end-(32*256) + 1 :end)=PowerNormalization(t5(:, end-(32*256) + 1 : end), 0.5);


temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(t1), NormalizeRowsUnit(t4), NormalizeRowsUnit(t5) ));
allDist{1,1}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(t2), NormalizeRowsUnit(t4), NormalizeRowsUnit(t5) ));
allDist{1,2}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(t3), NormalizeRowsUnit(t4), NormalizeRowsUnit(t5) ));
allDist{1,3}=temp * temp';


clear t1 t2 t3 t4 t5 


clear temp

all_clfsOut=cell(nEncoding,3);
all_accuracy=cell(nEncoding,3);

mean_all_accuracy=cell(nEncoding,1);



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

