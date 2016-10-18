%function  [all_accuracy, all_clfsOut]  = FisherFramework(typeFeature, normStrategy, d, cl, fsr, nPar, alpha)

% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;


%addpath('./../');%!!!!!!!
datasetName='UCF50';
typeFeature=@FEVidHmgDense;
normStrategy='ROOTSIFT';
d=72;
cl=256;
fsr=1;
nPar=5;
alpha=0.1;




% Parameter settings for descriptor extraction
clear descParam
descParam.Dataset=datasetName;
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;
descParam.pcaDim = d;
descParam.numClusters = cl;

descParam.NumBlocks = [3 3 2];


descParam.FrameSampleRate = fsr;
descParam.BlockSize = [8 8 6/fsr];

descParam.MediaType = 'Vid';
descParam.NumOr = 8;

%descParam.FrameSampleRate = 1;
%descParam.ColourSpace = colourSpace

sRow = [1 3];
sCol = [1 1];





descParam



vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=sprintf(DATAopts.videoPath, vocabularyIms{i});
end



[gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim);

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
fullPathVids=cell(size(vids));

for i=1:length(fullPathVids)
    fullPathVids{i}=sprintf(DATAopts.videoPath, vids{i});
end



    [tDesc] = MediaName2Descriptor(fullPathVids{1}, descParam, pcaMap);
    tDesc = tDesc'; 
    tFisher=mexFisherAssign(tDesc, gmmModelName)';
    
fisher1=zeros(length(vids), length(tFisher), 'like', tFisher);
fisher2=zeros(length(vids), length(tFisher), 'like', tFisher);
fisher3=zeros(length(vids), length(tFisher), 'like', tFisher);
fisher4=zeros(length(vids), length(tFisher), 'like', tFisher);

%parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(fullPathVids));
for i=1:length(fullPathVids)
    %fprintf('%d \n', i)
    % Extract descriptors
    if mod(i, 100)==0
        fprintf('%d ', i)%fprintf('%d \n', i)
    end

    [desc, info, descParamUsed] = MediaName2Descriptor(fullPathVids{i}, descParam, pcaMap);
    desc = desc';
    
        % Feature vector assignment with spatial pyramid
    featSpIdx = SpatialPyramidSeparationIdx(info, sRow, sCol)';
    
    
    fisher1(i, :)=mexFisherAssign(desc(:,featSpIdx(1,:)), gmmModelName)';
    fisher2(i, :)=mexFisherAssign(desc(:,featSpIdx(2,:)), gmmModelName)';
    fisher3(i, :)=mexFisherAssign(desc(:,featSpIdx(3,:)), gmmModelName)';
    fisher4(i, :)=mexFisherAssign(desc(:,featSpIdx(4,:)), gmmModelName)';
    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=1;
allDist=cell(1, nEncoding);

fisherAll=cat(2, fisher1,fisher2, fisher3, fisher4);


temp=NormalizeRowsUnit(PowerNormalization(fisherAll, alpha));
allDist{1}=temp * temp';

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

 bazeSavePath='/home/ionut/asustor_ionut/Data/results/cvpr2017/ucf50/';
 
 
    
saveName = [bazeSavePath 'clfsOut/'  DescParam2Name(descParam) '_fisherAll__all_accuracy.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = [bazeSavePath 'videoRep/' DescParam2Name(descParam) '__fisherAll.mat'];
 save(saveName2, '-v7.3', 'descParam', 'fisherAll');

