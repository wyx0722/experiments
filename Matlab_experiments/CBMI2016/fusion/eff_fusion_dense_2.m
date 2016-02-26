[vids, labs, groups] = GetVideosPlusLabels('Full');

%for real-time with VLAD
%HSM: (1, 6)
%HOF: (3, 2)
%HOG: (1, 6)
%MBHx: (3, 2)
%MBHy: (3, 2)


load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/frameSampleRate/FEVidHSMDenseBlockSize8_8_1_FrameSampleRate6MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters512pcaDim72_sRow3_VLAD_.mat');
hsm_VLAD01=NormalizeRowsUnit(PowerNormalization(VLADAll, 0.1));
hsm_VLAD01_noSP=NormalizeRowsUnit(PowerNormalization(VLADAll(:, 1:size(VLADAll, 2)/4), 0.1));

load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/frameSampleRate/FEVidHofDenseBlockSize8_8_3_FrameSampleRate2MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters512pcaDim72_sRow3_VLAD_.mat');
hof_VLAD01=NormalizeRowsUnit(PowerNormalization(VLADAll, 0.1));
hof_VLAD01_noSP=NormalizeRowsUnit(PowerNormalization(VLADAll(:, 1:size(VLADAll, 2)/4), 0.1));

load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/frameSampleRate/FEVidHogDenseBlockSize8_8_1_FrameSampleRate6MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters512pcaDim72_sRow3_VLAD_.mat');
hog_VLAD01=NormalizeRowsUnit(PowerNormalization(VLADAll, 0.1));
hog_VLAD01_noSP=NormalizeRowsUnit(PowerNormalization(VLADAll(:, 1:size(VLADAll, 2)/4), 0.1));

load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/frameSampleRate/FEVidMBHxDenseBlockSize8_8_3_FrameSampleRate2MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters512pcaDim72_sRow3_VLAD_.mat');
mbhx_VLAD01=NormalizeRowsUnit(PowerNormalization(VLADAll, 0.1));
mbhx_VLAD01_noSP=NormalizeRowsUnit(PowerNormalization(VLADAll(:, 1:size(VLADAll, 2)/4), 0.1));

load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/frameSampleRate/FEVidMBHyDenseBlockSize8_8_3_FrameSampleRate2MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters512pcaDim72_sRow3_VLAD_.mat');
mbhy_VLAD01=NormalizeRowsUnit(PowerNormalization(VLADAll, 0.1));
mbhy_VLAD01_noSP=NormalizeRowsUnit(PowerNormalization(VLADAll(:, 1:size(VLADAll, 2)/4), 0.1));





nEncoding=4;
allDist=cell(1, nEncoding);

ul_all5dense=NormalizeRowsUnit(cat(2,hof_VLAD01, hog_VLAD01, mbhx_VLAD01,mbhy_VLAD01,  hsm_VLAD01 ));
allDist{1}=ul_all5dense * ul_all5dense';
clear ul_all5dense


ul_hof_hog_mbhxy=NormalizeRowsUnit(cat(2,hof_VLAD01, hog_VLAD01, mbhx_VLAD01, mbhy_VLAD01));
allDist{2}=ul_hof_hog_mbhxy * ul_hof_hog_mbhxy';
clear ul_hof_hog_mbhxy


ul_all5dense_noSP=NormalizeRowsUnit(cat(2,hof_VLAD01_noSP, hog_VLAD01_noSP, mbhx_VLAD01_noSP,mbhy_VLAD01_noSP,  hsm_VLAD01_noSP ));
allDist{3}=ul_all5dense_noSP * ul_all5dense_noSP';
clear ul_all5dense_noSP

ul_hof_hog_mbhxy_noSP=NormalizeRowsUnit(cat(2,hof_VLAD01_noSP, hog_VLAD01_noSP, mbhx_VLAD01_noSP, mbhy_VLAD01_noSP));
allDist{4}=ul_hof_hog_mbhxy_noSP * ul_hof_hog_mbhxy_noSP';
clear ul_hof_hog_mbhxy_noSP

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
acc4=mean(mean(cat(2, all_accuracy{4}{:})))






