global DATAopts;
DATAopts = UCFInit;
addpath('./../..')
addpath('./..')

[vids, labs, groups] = GetVideosPlusLabels('Full');

alpha_handF=0.1
alpha_deepF=0.5


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/videoRep/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureHOF_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim54spClusters32.mat']
load(pathF);
hof_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
hof_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
hof_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/videoRep/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureHOG_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
hog_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
hog_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
hog_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/videoRep/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureMBHx_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
mbhx_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
mbhx_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
mbhx_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/videoRep/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureMBHy_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
mbhy_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
mbhy_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
mbhy_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/videoRep/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF50Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19pcaDim256spClusters32.mat']
load(pathF);
spVGG19_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_deepF));
spVGG19_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_deepF));
spVGG19_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_deepF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/videoRep/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF50Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit1VGG16pcaDim256spClusters32.mat']
load(pathF);
tempVGG16_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_deepF));
tempVGG16_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_deepF));
tempVGG16_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_deepF));


clear  v256 v512 spV32 descParam nDesc



%% Do classification
nEncoding=9;%!!!!!!!!change!!!!!!
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(cat(2,hof_norm_v256, hog_norm_v256, mbhx_norm_v256, mbhy_norm_v256));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(cat(2,hof_norm_v512, hog_norm_v512, mbhx_norm_v512, mbhy_norm_v512));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(cat(2,hof_norm_spV32, hog_norm_spV32, mbhx_norm_spV32, mbhy_norm_spV32));
allDist{3}=temp * temp';


temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v256, tempVGG16_norm_v256));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v512, tempVGG16_norm_v512));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(cat(2,spVGG19_norm_spV32, tempVGG16_norm_spV32));
allDist{6}=temp * temp';


temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v256, tempVGG16_norm_v256, hof_norm_v256, hog_norm_v256, mbhx_norm_v256, mbhy_norm_v256));
allDist{7}=temp * temp';

temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v512, tempVGG16_norm_v512, hof_norm_v512, hog_norm_v512, mbhx_norm_v512, mbhy_norm_v512));
allDist{8}=temp * temp';

temp=NormalizeRowsUnit(cat(2,spVGG19_norm_spV32, tempVGG16_norm_spV32,hof_norm_spV32, hog_norm_spV32, mbhx_norm_spV32, mbhy_norm_spV32));
allDist{9}=temp * temp';

clear temp


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

mean(perGroupAccuracy)

end
delete(gcp('nocreate'))


finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    finalAcc(j)=mean(mean(cat(2, all_accuracy{j}{:}), 2));
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));  
end