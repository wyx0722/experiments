
%load('/data/ionut/rezults/mtap2017/FEVidHogDenseBlockSize8_8_3_FrameSampleRate2MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72.mat');
%SD_VLADAll
%% Do classification

global DATAopts;
DATAopts = UCFInit;

[vids, labs, groups] = GetVideosPlusLabels('Full');



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


