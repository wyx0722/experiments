
addpath('./../');%!!!!!!!
addpath('./../../');%!!!!!!!

global DATAopts;
DATAopts = HMDB51Init;
[allVids, labs, splits] = GetVideosPlusLabels();


alpha=0.5;
datasetName='HMDB51'





alpha_deepF=0.5;
alpha_handF=0.1;


bazeDir='/home/ionut/asustor_ionut/Data/results/cvpr2017/hmdb51/clusters/videoRep/'






name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerconv5bMediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetC3DpcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl256.mat']
load(name);
C3D_sp32cl256=sp32cl256;
C3D_sp32cl256(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256(:, end-(32*256) + 1 : end), alpha_deepF);
C3D_sp32cl256_descParam=descParam;


name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetSpVGG19pcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl256.mat']
load(name);
SpVGG19_sp32cl256=sp32cl256;
SpVGG19_sp32cl256(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256(:, end-(32*256) + 1 : end), alpha_deepF);
SpVGG19_sp32cl256_descParam=descParam;

name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetTempSplit1VGG16pcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl256.mat']
load(name);
TempSplit1VGG16_sp32cl256=sp32cl256;
TempSplit1VGG16_sp32cl256(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256(:, end-(32*256) + 1 : end), alpha_deepF);
TempSplit1VGG16_sp32cl256_descParam=descParam;

clear sp32cl256







nEncoding=6;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(C3D_sp32cl256); allDist{1}=temp * temp';
temp=NormalizeRowsUnit(SpVGG19_sp32cl256); allDist{2}=temp * temp';
temp=NormalizeRowsUnit(TempSplit1VGG16_sp32cl256); allDist{3}=temp * temp';

temp=NormalizeRowsUnit(C3D_sp32cl256(:, 1:256*512)); allDist{4}=temp * temp';
temp=NormalizeRowsUnit(SpVGG19_sp32cl256(:, 1:256*512)); allDist{5}=temp * temp';
temp=NormalizeRowsUnit(TempSplit1VGG16_sp32cl256(:, 1:256*512)); allDist{6}=temp * temp';

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



bazeSavePath='/home/ionut/asustor_ionut/Data/results/cvpr2017/forFusion_state_of_the_art/hmdb51/'

clfsOut_sp32cl256PCA0 = all_clfsOut(1);
acc_sp32cl256PCA0 = all_accuracy(1);
intructions_compute_acc='mean((acc_sp32cl256PCA0{1}{1} + acc_sp32cl256PCA0{1}{2} + acc_sp32cl256PCA0{1}{3})./3)';
saveName = [bazeSavePath '/clfsOut/'  DescParam2Name(C3D_sp32cl256_descParam) '__clfsOut_sp32cl256PCA0.mat']
save(saveName, '-v7.3', 'C3D_sp32cl256_descParam', 'clfsOut_sp32cl256PCA0', 'acc_sp32cl256PCA0', 'intructions_compute_acc');

clfsOut_sp32cl256PCA0 = all_clfsOut(2);
acc_sp32cl256PCA0 = all_accuracy(2);
intructions_compute_acc='mean((acc_sp32cl256PCA0{1}{1} + acc_sp32cl256PCA0{1}{2} + acc_sp32cl256PCA0{1}{3})./3)';
saveName = [bazeSavePath '/clfsOut/'  DescParam2Name(SpVGG19_sp32cl256_descParam) '__clfsOut_sp32cl256PCA0.mat']
save(saveName, '-v7.3', 'SpVGG19_sp32cl256_descParam', 'clfsOut_sp32cl256PCA0', 'acc_sp32cl256PCA0', 'intructions_compute_acc');

clfsOut_sp32cl256PCA0 = all_clfsOut(3);
acc_sp32cl256PCA0 = all_accuracy(3);
intructions_compute_acc='mean((acc_sp32cl256PCA0{1}{1} + acc_sp32cl256PCA0{1}{2} + acc_sp32cl256PCA0{1}{3})./3)';
saveName = [bazeSavePath '/clfsOut/'  DescParam2Name(TempSplit1VGG16_sp32cl256_descParam) '__clfsOut_sp32cl256PCA0.mat']
save(saveName, '-v7.3', 'TempSplit1VGG16_sp32cl256_descParam', 'clfsOut_sp32cl256PCA0', 'acc_sp32cl256PCA0', 'intructions_compute_acc');




