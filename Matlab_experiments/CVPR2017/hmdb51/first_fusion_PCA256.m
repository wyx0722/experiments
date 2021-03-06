
addpath('./../');%!!!!!!!

global DATAopts;
DATAopts = HMDB51Init;
[allVids, labs, splits] = GetVideosPlusLabels();


alpha=0.5;
datasetName='HMDB51'

name='/home/ionut/asustor_ionut/Data/results/cvpr2017/beginning_rezults/FEVid_deepFeaturesClusters256_319_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit1VGG16pcaDim256spClusters32__v256__m256__st_vlmpf32.mat'
load(name);

v256_PCA256_temp=v256;
m256_PCA256_temp=m256;
st_vlmpf32_PCA256_temp=st_vlmpf32;


name='/home/ionut/asustor_ionut/Data/results/cvpr2017/beginning_rezults/FEVid_deepFeaturesClusters256_319_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19pcaDim256spClusters32__v256__m256__st_vlmpf32.mat'
load(name);

v256_PCA256_sp=v256;
m256_PCA256_sp=m256;
st_vlmpf32_PCA256_sp=st_vlmpf32;

name='/home/ionut/asustor_ionut/Data/results/cvpr2017/beginning_rezults/FEVid_deepFeaturesClusters256_319_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormalisationNonenetC3DpcaDim256spClusters32__v256__m256__st_vlmpf32.mat'
load(name);

v256_PCA256_c3d=v256;
m256_PCA256_c3d=m256;
st_vlmpf32_PCA256_c3d=st_vlmpf32;



nEncoding=3;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(v256_PCA256_temp, alpha)), NormalizeRowsUnit(PowerNormalization(v256_PCA256_sp, alpha)), ...
    NormalizeRowsUnit(PowerNormalization(v256_PCA256_c3d, alpha)) ) );
allDist{1}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(m256_PCA256_temp, alpha)), NormalizeRowsUnit(PowerNormalization(m256_PCA256_sp, alpha)), ...
    NormalizeRowsUnit(PowerNormalization(m256_PCA256_c3d, alpha)) ) );
allDist{2}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(st_vlmpf32_PCA256_temp, alpha)), NormalizeRowsUnit(PowerNormalization(st_vlmpf32_PCA256_sp, alpha)), ...
    NormalizeRowsUnit(PowerNormalization(st_vlmpf32_PCA256_c3d, alpha)) ) );
allDist{3}=temp * temp';

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