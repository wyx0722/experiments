global DATAopts;
DATAopts = HMDB51Init;
addpath('./../..')
addpath('./..')

[allVids, labs, splits] = GetVideosPlusLabels();

alpha_handF=0.1
alpha_deepF=0.5


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/videoRep/' 'FEVid_IDTClusters64_128_256_512_DatasetHMBD51IDTfeatureHOF_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim54spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
hof_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
hof_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
hof_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/videoRep/' 'FEVid_IDTClusters64_128_256_512_DatasetHMBD51IDTfeatureHOG_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
hog_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
hog_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
hog_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/videoRep/' 'FEVid_IDTClusters64_128_256_512_DatasetHMBD51IDTfeatureMBHx_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
mbhx_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
mbhx_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
mbhx_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/videoRep/' 'FEVid_IDTClusters64_128_256_512_DatasetHMBD51IDTfeatureMBHy_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
mbhy_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
mbhy_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
mbhy_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/videoRep/' 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMBD51Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19pcaDim256spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
spVGG19_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_deepF));
spVGG19_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_deepF));
spVGG19_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_deepF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/videoRep/' 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMBD51Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit1VGG16pcaDim256spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
tempVGG16_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_deepF));
tempVGG16_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_deepF));
tempVGG16_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_deepF));


clear v64 v128 v256 v512 spV2 spV4 spV8 spV16 spV32 spV64 spV128 spV255 descParam nDesc



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


%each row for the cell represents the results for all 3 splits
all_clfsOut=cell(nEncoding,3);
all_accuracy=cell(nEncoding,3);
mean_all_clfsOut=cell(nEncoding,1);
mean_all_accuracy=cell(nEncoding,1);

cRange = 100;
nReps = 1;
nFolds = 3;


parpool(9);
parfor k=1:nEncoding
    k
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI  = splits(:,i) == 2;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        trainDist = allDist{k}(trainI, trainI);
        testDist = allDist{k}(testI, trainI);
        

        [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        accuracy = ClassificationAccuracy(clfsOut, testLabs);
        fprintf('accuracy: %.3f\n', accuracy);

        all_clfsOut{k,i}=clfsOut;
        all_accuracy{k,i}=accuracy;
    end
end

delete(gcp('nocreate'))


finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    mean_all_clfsOut{j}=(all_clfsOut{j,1} + all_clfsOut{j,2} + all_clfsOut{j,3})./3;
    mean_all_accuracy{j}=(all_accuracy{j,1} + all_accuracy{j,2} + all_accuracy{j,3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end