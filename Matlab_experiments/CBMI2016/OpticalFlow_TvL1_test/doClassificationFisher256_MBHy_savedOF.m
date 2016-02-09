% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVidMBHy_savedOpticalFlow;
descParam.BlockSize = [8 8 6];
descParam.NumBlocks = [3 3 2];
descParam.MediaType = 'savedOF';
descParam.NumOr = 8;
%descParam.FrameSampleRate = 1;
%descParam.ColourSpace = colourSpace

sRow = [1 3];
sCol = [1 1];


descParam.pcaDim = 72;
descParam.numClusters = 256;

descParam


%%%%%%%%%%
bazePathFeatures='/home/ionut/Data/UCF50_tvL1_OpticalFlow/Videos/'; %change

vocabularyIms = GetVideosPlusLabels('smallEnd');
vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i))];
end

[gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim);

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');

pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i))];
end


    [tDesc] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
    tDesc = tDesc'; 
    tFisher=mexFisherAssign(tDesc, gmmModelName)';
    
fisher1=zeros(length(vids), length(tFisher), 'like', tFisher);
fisher2=zeros(length(vids), length(tFisher), 'like', tFisher);
fisher3=zeros(length(vids), length(tFisher), 'like', tFisher);
fisher4=zeros(length(vids), length(tFisher), 'like', tFisher);

parpool(5);
% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
parfor i=1:length(pathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
    desc = desc';
    
        % Feature vector assignment with spatial pyramid
    featSpIdx = SpatialPyramidSeparationIdx(info, sRow, sCol)';
    fisherVT = cell(1,size(featSpIdx,1));
    
    fisher1(i, :)=mexFisherAssign(desc(:,featSpIdx(1,:)), gmmModelName)';
    fisher2(i, :)=mexFisherAssign(desc(:,featSpIdx(2,:)), gmmModelName)';
    fisher3(i, :)=mexFisherAssign(desc(:,featSpIdx(3,:)), gmmModelName)';
    fisher4(i, :)=mexFisherAssign(desc(:,featSpIdx(4,:)), gmmModelName)';
    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');

delete(gcp('nocreate'))


%% Do classification

nEncoding=2;
allDist=cell(1, nEncoding);

fisherAll=cat(2, fisher1,fisher2, fisher3, fisher4);


n_fisherAll=NormalizeRowsUnit(PowerNormalization(fisherAll, 0.5));
allDist{1}=n_fisherAll * n_fisherAll';
clear n_fisherAll

n_fisher1=NormalizeRowsUnit(PowerNormalization(fisher1, 0.5));
allDist{2}=n_fisher1 * n_fisher1';
clear n_fisher1

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


parpool(13);
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

% v1_meanAcc=mean(mean(cat(2, all_accuracy{1}{:})));
% v2_meanAcc=mean(mean(cat(2, all_accuracy{2}{:})));

saveName = ['/home/ionut/Data/results/CBMI2015_rezults/' 'clfsOut/' 'savedOF/' DescParam2Name(descParam) '_sRow3_Fisher_.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = ['/home/ionut/Data/results/CBMI2015_rezults/' 'videoRep/' 'savedOF/' DescParam2Name(descParam) '_sRow3_Fisher_.mat'];
 save(saveName2, '-v7.3', 'fisherAll');
