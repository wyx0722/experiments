[vids, labs, groups] = GetVideosPlusLabels('Full');




load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/frameSampleRate/FEVidHogDenseBlockSize8_8_1_FrameSampleRate6MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72_sRow3_Fisher_.mat');
hog_fisher01=NormalizeRowsUnit(PowerNormalization(fisherAll, 0.1));
hog_fisher01_noSP=NormalizeRowsUnit(PowerNormalization(fisherAll(:, 1:size(fisherAll, 2)/4), 0.1));

load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/frameSampleRate/FEVidHSMDenseBlockSize8_8_1_FrameSampleRate6MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters512pcaDim72_sRow3_VLAD_.mat');
hsm_VLAD01=NormalizeRowsUnit(PowerNormalization(VLADAll, 0.1));
hsm_VLAD01_noSP=NormalizeRowsUnit(PowerNormalization(VLADAll(:, 1:size(VLADAll, 2)/4), 0.1));

load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/frameSampleRate/FEVidHofDenseBlockSize8_8_3_FrameSampleRate2MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters512pcaDim72_sRow3_VLAD_.mat');
hof_VLAD01=NormalizeRowsUnit(PowerNormalization(VLADAll, 0.1));
hof_VLAD01_noSP=NormalizeRowsUnit(PowerNormalization(VLADAll(:, 1:size(VLADAll, 2)/4), 0.1));




nEncoding=3;
allDist=cell(1, nEncoding);

ul_alldense=NormalizeRowsUnit(cat(2,hsm_VLAD01, hog_fisher01, hof_VLAD01));
allDist{1}=ul_alldense * ul_alldense';
clear ul_alldense


ul_alldense_noSP=NormalizeRowsUnit(cat(2,hsm_VLAD01_noSP, hog_fisher01_noSP, hof_VLAD01_noSP));
allDist{2}=ul_alldense_noSP * ul_alldense_noSP';
clear ul_all5dense_noSP

ul_hog_hsm=NormalizeRowsUnit(cat(2,hsm_VLAD01, hog_fisher01));
allDist{3}=ul_hog_hsm * ul_hog_hsm';
clear ul_hog_hsm

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
acc3=mean(mean(cat(2, all_accuracy{3}{:})))






