
% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;
addpath('/home/ionut/experiments/Matlab_experiments/IDT_UCF50/')

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_IDT;
descParam.MediaType = 'IDT';
descParam.IDTfeature='HOF';
descParam.Normalisation='ROOTSIFT'; % L2 or 'ROOTSIFT'


if strcmp(descParam.IDTfeature,'HOF')
    sizeDesc=108;
    
elseif  strcmp(descParam.IDTfeature,'HOG') || strcmp(descParam.IDTfeature,'MBHx') || strcmp(descParam.IDTfeature,'MBHy')  
    sizeDesc=96;   
end

% pcaDim & vocabulary size
descParam.pcaDim = sizeDesc/2;
descParam.numClusters = 256;


descParam

%%%%%%%%%%
bazePathFeatures='/home/ionut/Features/Features/UCF50/IDT/Videos/'; %change

vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=sprintf(bazePathFeatures, vocabularyIms{i});
end



[gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim);

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
fullPathVids=cell(size(vids));

for i=1:length(fullPathVids)
    fullPathVids{i}=sprintf(bazePathFeatures, vids{i});
end



    [tDesc] = MediaName2Descriptor(fullPathVids{1}, descParam, pcaMap);
    tDesc = tDesc'; 
    tFisher=mexFisherAssign(tDesc, gmmModelName)';
    
fisherV=zeros(length(vids), length(tFisher), 'like', tFisher);


parpool(9);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(fullPathVids));
parfor i=1:length(fullPathVids)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(fullPathVids{i}, descParam, pcaMap);
    desc = desc';
    
 
    
    
    fisherV(i, :)=mexFisherAssign(desc, gmmModelName)';
   
    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=2;
allDist=cell(1, nEncoding);



n_fisherV=NormalizeRowsUnit(PowerNormalization(fisherV, 0.5));
allDist{1}=n_fisherV * n_fisherV';
clear n_fisherV

n_fisherV2=NormalizeRowsUnit(PowerNormalization(fisherV, 0.1));
allDist{2}=n_fisherV2 * n_fisherV2';
clear n_fisherV2

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


for k=1:nEncoding

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

k
perGroupAccuracy = mean(cat(2, accuracy{:}))'

end

delete(gcp('nocreate'))




acc1=mean(mean(cat(2, all_accuracy{1}{:})))
acc2=mean(mean(cat(2, all_accuracy{2}{:})))




saveName = ['/home/ionut/Data/results/CBMI2015_rezults/' 'clfsOut/' 'IDT/' DescParam2Name(descParam) '_Fisher_.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = ['/home/ionut/Data/results/CBMI2015_rezults/' 'videoRep/' 'IDT/' DescParam2Name(descParam) '_Fisher_.mat'];
 save(saveName2, '-v7.3', 'fisherV');
