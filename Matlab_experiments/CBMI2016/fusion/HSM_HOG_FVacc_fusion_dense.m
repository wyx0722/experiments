[vids, labs, groups] = GetVideosPlusLabels('Full');


load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/frameSampleRate/FEVidHSMDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72_sRow3_Fisher_.mat');
hsm_fisher01=NormalizeRowsUnit(PowerNormalization(fisherAll, 0.1));
hsm_fisher01_noSP=NormalizeRowsUnit(PowerNormalization(fisherAll(:, 1:size(fisherAll, 2)/4), 0.1));


load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/frameSampleRate/FEVidHogDenseBlockSize8_8_1_FrameSampleRate6MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72_sRow3_Fisher_.mat');
hog_fisher01=NormalizeRowsUnit(PowerNormalization(fisherAll, 0.1));
hog_fisher01_noSP=NormalizeRowsUnit(PowerNormalization(fisherAll(:, 1:size(fisherAll, 2)/4), 0.1));


nEncoding=2;
allDist=cell(1, nEncoding);

hog_hof_dense=NormalizeRowsUnit(cat(2,hsm_fisher01, hog_fisher01));
allDist{1}=hog_hof_dense * hog_hof_dense';
clear hog_hof_dense


hog_hof_dense_noSP=NormalizeRowsUnit(cat(2,hsm_fisher01_noSP, hog_fisher01_noSP));
allDist{2}=hog_hof_dense_noSP * hog_hof_dense_noSP';
clear hog_hof_dense_noSP



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


acc1=mean(mean(cat(2, all_accuracy{1}{:})))
acc2=mean(mean(cat(2, all_accuracy{2}{:})))






