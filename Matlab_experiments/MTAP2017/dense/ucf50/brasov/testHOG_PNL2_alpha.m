
%load('/data/ionut/rezults/mtap2017/FEVidHogDenseBlockSize8_8_3_FrameSampleRate2MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72.mat');
%SD_VLADAll
%% Do classification

alpha=[0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1];
nEncoding=9;
allDist=cell(1, nEncoding);

for i=1:length(alpha)
    if intraL2==1
        temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(SD_VLADAll, 72), alpha(i)));
    else
       temp=NormalizeRowsUnit(PowerNormalization(SD_VLADAll, alpha(i)));
    end
    
    allDist{i}=temp * temp';
        
end


all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


rezPath='/data/ionut/rezults/mtap2017/'
fileName=[rezPath  'resultsHOG_PNL2_SD_VLADAll_intraPNL2_'];

fileName=sprintf('%s%g.txt', fileName, intraL2)

fileID=fopen(fileName, 'a');
fprintf(fileID, 'FEVidHogDenseBlockSize8_8_3_FrameSampleRate2MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72  \n\n' );
fclose(fileID);


parpool(5);
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
mean(mean(cat(2, accuracy{:})))

fileID=fopen(fileName, 'a');
fprintf(fileID, 'PNL2 norm before classification -> alpha=%.2f --> %.3f  \n', ...
             alpha(k), mean(mean(cat(2, all_accuracy{k}{:}))) );
fclose(fileID);

end

delete(gcp('nocreate'))


    
saveName = [rezPath 'clfsOut/' fileName '.mat'];
save(saveName, '-v7.3', 'all_clfsOut', 'all_accuracy');


