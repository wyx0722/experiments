
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





name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetSpVGG19pcaDim0spClusters2_4_8_16_32_64_128_256___sp32cl256.mat']
load(name);
%SpVGG19_sp32cl256=sp32cl256;
sp32cl256(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256(:, end-(32*256) + 1 : end), alpha_deepF);
n_SpVGG19_sp32cl256=NormalizeRowsUnit(sp32cl256);






nEncoding=1;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(n_SpVGG19_sp32cl256(:, 1:256*512));
allDist{1}=temp * temp';

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


