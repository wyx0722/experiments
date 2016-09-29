
% addpath('./../');%!!!!!!!
% 
% global DATAopts;
% DATAopts = HMDB51Init;
% [allVids, labs, splits] = GetVideosPlusLabels();
% 
% 
% alpha_deep=0.5;
% alpha_idt=0.1;
% datasetName='HMDB51'
% 
% name='/home/ionut/asustor_ionut/Data/results/cvpr2017/beginning_rezults/FEVid_deepFeaturesClusters256_319_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit1VGG16pcaDim0spClusters32__v256__m256__st_vlmpf32.mat'
% load(name);
% 
% v256_noPCA_temp=v256;
% m256_noPCA_temp=m256;
% st_vlmpf32_noPCA_temp=st_vlmpf32;
% 
% 
% name='/home/ionut/asustor_ionut/Data/results/cvpr2017/beginning_rezults/FEVid_deepFeaturesClusters256_319_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19pcaDim0spClusters32__v256__m256__st_vlmpf32.mat'
% load(name);
% 
% v256_noPCA_sp=v256;
% m256_noPCA_sp=m256;
% st_vlmpf32_noPCA_sp=st_vlmpf32;
% 
% name='/home/ionut/asustor_ionut/Data/results/cvpr2017/beginning_rezults/FEVid_deepFeaturesClusters256_319_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormalisationNonenetC3DpcaDim0spClusters32__v256__m256__st_vlmpf32.mat'
% load(name);
% 
% v256_noPCA_c3d=v256;
% m256_noPCA_c3d=m256;
% st_vlmpf32_noPCA_c3d=st_vlmpf32;
% 
% 
% name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/iDT/FEVid_IDTDatasetHMDB51IDTfeatureHOFMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim54fisherVector.mat'
% load(name);
% hof_idt=fisherVector;
% 
% name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/iDT/FEVid_IDTDatasetHMDB51IDTfeatureHOGMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat'
% load(name);
% hog_idt=fisherVector;
% 
% name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/iDT/FEVid_IDTDatasetHMDB51IDTfeatureMBHxMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat'
% load(name);
% mbhx_idt=fisherVector;
% 
% name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/iDT/FEVid_IDTDatasetHMDB51IDTfeatureMBHyMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat'
% load(name);
% mbhy_idt=fisherVector;
% 
% clear fisherVector v256 m256 st_vlmpf32
% fprintf('\nDone! load features \n');

 alpha_deep=1;
 alpha_idt=0.1;
 
 
nEncoding=3;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(v256_noPCA_temp, alpha_deep)), NormalizeRowsUnit(PowerNormalization(v256_noPCA_sp, alpha_deep)), ...
    NormalizeRowsUnit(PowerNormalization(v256_noPCA_c3d, alpha_deep)), ...
    NormalizeRowsUnit(PowerNormalization(hof_idt, alpha_idt)), NormalizeRowsUnit(PowerNormalization(hog_idt, alpha_idt)), ...
    NormalizeRowsUnit(PowerNormalization(mbhx_idt, alpha_idt)),NormalizeRowsUnit(PowerNormalization(mbhy_idt, alpha_idt)) ) );
allDist{1}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(m256_noPCA_temp, alpha_deep)), NormalizeRowsUnit(PowerNormalization(m256_noPCA_sp, alpha_deep)), ...
    NormalizeRowsUnit(PowerNormalization(m256_noPCA_c3d, alpha_deep)) , ...
    NormalizeRowsUnit(PowerNormalization(hof_idt, alpha_idt)), NormalizeRowsUnit(PowerNormalization(hog_idt, alpha_idt)), ...
    NormalizeRowsUnit(PowerNormalization(mbhx_idt, alpha_idt)),NormalizeRowsUnit(PowerNormalization(mbhy_idt, alpha_idt)) ) );
allDist{2}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(st_vlmpf32_noPCA_temp, alpha_deep)), NormalizeRowsUnit(PowerNormalization(st_vlmpf32_noPCA_sp, alpha_deep)), ...
    NormalizeRowsUnit(PowerNormalization(st_vlmpf32_noPCA_c3d, alpha_deep)), ...
    NormalizeRowsUnit(PowerNormalization(hof_idt, alpha_idt)), NormalizeRowsUnit(PowerNormalization(hog_idt, alpha_idt)), ...
    NormalizeRowsUnit(PowerNormalization(mbhx_idt, alpha_idt)),NormalizeRowsUnit(PowerNormalization(mbhy_idt, alpha_idt)) ) );
allDist{3}=temp * temp';

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



%%%
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
end

delete(gcp('nocreate'))
%%%%

finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end