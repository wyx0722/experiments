
addpath('./../');%!!!!!!!
addpath('./../../');%!!!!!!!

global DATAopts;
DATAopts = HMDB51Init;
[allVids, labs, splits] = GetVideosPlusLabels();


alpha=0.5;
datasetName='HMDB51'

bazeDir='/home/ionut/asustor_ionut/Data/results/cvpr2017/hmdb51/clusters/indv_cl/videoRep/'

name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerconv5bMediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetC3DpcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl64.mat']
load(name);
C3D_sp32cl64=sp32cl64;

name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetSpVGG19pcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl64.mat']
load(name);
SpVGG19_sp32cl64=sp32cl64;

name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetTempSplit1VGG16pcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl64.mat']
load(name);
TempSplit1VGG16_sp32cl64=sp32cl64;

clear sp32cl64


name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerconv5bMediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetC3DpcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl128.mat']
load(name);
C3D_sp32cl128=sp32cl128;

name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetSpVGG19pcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl128.mat']
load(name);
SpVGG19_sp32cl128=sp32cl128;

name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetTempSplit1VGG16pcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl128.mat']
load(name);
TempSplit1VGG16_sp32cl128=sp32cl128;

clear sp32cl128





nEncoding=2;
allDist=cell(1, nEncoding);


t1=TempSplit1VGG16_sp32cl64;
t1(:, end-(32*64) + 1 :end)=PowerNormalization(t1(:, end-(32*64) + 1 : end), 0.5);

t2=SpVGG19_sp32cl64;
t2(:, end-(32*64) + 1 :end)=PowerNormalization(t2(:, end-(32*64) + 1 : end), 0.5);

t3=C3D_sp32cl64;
t3(:, end-(32*64) + 1 :end)=PowerNormalization(t3(:, end-(32*64) + 1 : end), 0.5);

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(t1), ...
    NormalizeRowsUnit(t2), NormalizeRowsUnit(t3) ));
allDist{1}=temp * temp';


t1=TempSplit1VGG16_sp32cl128;
t1(:, end-(32*128) + 1 :end)=PowerNormalization(t1(:, end-(32*128) + 1 : end), 0.5);

t2=SpVGG19_sp32cl128;
t2(:, end-(32*128) + 1 :end)=PowerNormalization(t2(:, end-(32*128) + 1 : end), 0.5);

t3=C3D_sp32cl128;
t3(:, end-(32*128) + 1 :end)=PowerNormalization(t3(:, end-(32*128) + 1 : end), 0.5);

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(t1), ...
    NormalizeRowsUnit(t2), NormalizeRowsUnit(t3) ));
allDist{2}=temp * temp';

clear t1 t2 t3
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