addpath('./../');%!!!!!!!

global DATAopts;
DATAopts = UCFInit;


[vids, labs, groups] = GetVideosPlusLabels('Full');




baze_dir='/home/ionut/asustor_ionut/Data/results/mtap2017/ucf50/videoRep/VLADbased/'


load_name=[baze_dir 'FEVidHogDenseBlockSize8_8_1_FrameSampleRate6MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72_SD_VLAD__SD_VLADAll.mat']
load(load_name);
hog_sd_vlad_6=SD_VLADAll;
clear SD_VLADAll


load_name=[baze_dir 'FEVidHmgDenseBlockSize8_8_1_FrameSampleRate6MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72_SD_VLAD__SD_VLADAll.mat']
load(load_name);
hmg_sd_vlad_6=SD_VLADAll;
clear SD_VLADAll


load_name=[baze_dir 'FEVidHofDenseBlockSize8_8_2_FrameSampleRate3MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72_SD_VLAD__SD_VLADAll.mat']
load(load_name);
hof_sd_vlad_3=SD_VLADAll;
clear SD_VLADAll

load_name=[baze_dir 'FEVidMBHxDenseBlockSize8_8_2_FrameSampleRate3MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72_SD_VLAD__SD_VLADAll.mat']
load(load_name);
mbhx_sd_vlad_3=SD_VLADAll;
clear SD_VLADAll


load_name=[baze_dir 'FEVidMBHyDenseBlockSize8_8_2_FrameSampleRate3MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72_SD_VLAD__SD_VLADAll.mat']
load(load_name);
mbhy_sd_vlad_3=SD_VLADAll;
clear SD_VLADAll



%% Do classification

nEncoding=1;
allDist=cell(1, nEncoding);



temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(hog_sd_vlad_6, 72), 0.5)), ...
       NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(hmg_sd_vlad_6, 72), 0.5)), ...
       NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(hof_sd_vlad_3, 72), 0.5)), ...
       NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(mbhx_sd_vlad_3, 72), 0.5)), ...
       NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(mbhy_sd_vlad_3, 72), 0.5)) ...
       ));
allDist{1}=temp * temp';


all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


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

end

delete(gcp('nocreate'))



for j=1:nEncoding
    
    fprintf('Encoding %d --> MAcc: %.3f \n', j, mean(mean(cat(2, all_accuracy{j}{:}))));
end