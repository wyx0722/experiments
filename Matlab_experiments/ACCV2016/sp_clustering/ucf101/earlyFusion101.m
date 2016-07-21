global DATAopts;
DATAopts = UCF101Init;
addpath('./../..')
addpath('./..')

[allVids, labs, splits] = GetVideosPlusLabels('Challenge');

alpha_handF=0.1
alpha_deepF=0.5


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/videoRep/' 'FEVid_IDTClusters256_512_DatasetUCF101IDTfeatureHOF_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim54spClusters32.mat']
load(pathF);
hof_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
hof_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
hof_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/videoRep/' 'FEVid_IDTClusters256_512_DatasetUCF101IDTfeatureHOG_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
hog_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
hog_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
hog_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/videoRep/' 'FEVid_IDTClusters256_512_DatasetUCF101IDTfeatureMBHx_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
mbhx_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
mbhx_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
mbhx_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/videoRep/' 'FEVid_IDTClusters256_512_DatasetUCF101IDTfeatureMBHy_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
mbhy_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_handF));
mbhy_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_handF));
mbhy_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_handF));


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/videoRep/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF101Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19pcaDim256spClusters32.mat']
load(pathF);
spVGG19_norm_v256 = NormalizeRowsUnit(PowerNormalization(v256, alpha_deepF));
spVGG19_norm_v512 = NormalizeRowsUnit(PowerNormalization(v512, alpha_deepF));
spVGG19_norm_spV32 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_deepF));


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/videoRep/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF101Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit1VGG16pcaDim256spClusters32.mat']
load(pathF);
tempVGG16_norm_v256_split1 = NormalizeRowsUnit(PowerNormalization(v256, alpha_deepF));
tempVGG16_norm_v512_split1 = NormalizeRowsUnit(PowerNormalization(v512, alpha_deepF));
tempVGG16_norm_spV32_split1 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_deepF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/videoRep/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF101Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit2VGG16pcaDim256spClusters32.mat']
load(pathF);
tempVGG16_norm_v256_split2 = NormalizeRowsUnit(PowerNormalization(v256, alpha_deepF));
tempVGG16_norm_v512_split2 = NormalizeRowsUnit(PowerNormalization(v512, alpha_deepF));
tempVGG16_norm_spV32_split2 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_deepF));

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/videoRep/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF101Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit3VGG16pcaDim256spClusters32.mat']
load(pathF);
tempVGG16_norm_v256_split3 = NormalizeRowsUnit(PowerNormalization(v256, alpha_deepF));
tempVGG16_norm_v512_split3 = NormalizeRowsUnit(PowerNormalization(v512, alpha_deepF));
tempVGG16_norm_spV32_split3 = NormalizeRowsUnit(PowerNormalization(spV32, alpha_deepF));


clear  v256 v512 spV32 descParam nDesc



%% Do classification
nEncoding=9;%!!!!!!!!change!!!!!!
allDist=cell(nEncoding, 3);

temp=NormalizeRowsUnit(cat(2,hof_norm_v256, hog_norm_v256, mbhx_norm_v256, mbhy_norm_v256));
allDist{1}{1}=temp * temp'; allDist{1}{2}=allDist{1}{1}; allDist{1}{3}=allDist{1}{1};

temp=NormalizeRowsUnit(cat(2,hof_norm_v512, hog_norm_v512, mbhx_norm_v512, mbhy_norm_v512));
allDist{2}{1}=temp * temp'; allDist{2}{2}=allDist{2}{1}; allDist{2}{3}=allDist{2}{1};

temp=NormalizeRowsUnit(cat(2,hof_norm_spV32, hog_norm_spV32, mbhx_norm_spV32, mbhy_norm_spV32));
allDist{3,1}=temp * temp'; allDist{3}{2}=allDist{3}{1}; allDist{3}{3}=allDist{3}{1};


temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v256, tempVGG16_norm_v256_split1));
allDist{4,1}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v256, tempVGG16_norm_v256_split2));
allDist{4,2}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v256, tempVGG16_norm_v256_split3));
allDist{4,3}=temp * temp';


temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v512, tempVGG16_norm_v512_split1));
allDist{5,1}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v512, tempVGG16_norm_v512_split2));
allDist{5,2}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v512, tempVGG16_norm_v512_split3));
allDist{5,3}=temp * temp';

temp=NormalizeRowsUnit(cat(2,spVGG19_norm_spV32, tempVGG16_norm_spV32_split1));
allDist{6,1}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_spV32, tempVGG16_norm_spV32_split2));
allDist{6,2}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_spV32, tempVGG16_norm_spV32_split3));
allDist{6,3}=temp * temp';




temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v256, tempVGG16_norm_v256_split1, hof_norm_v256, hog_norm_v256, mbhx_norm_v256, mbhy_norm_v256));
allDist{7,1}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v256, tempVGG16_norm_v256_split2, hof_norm_v256, hog_norm_v256, mbhx_norm_v256, mbhy_norm_v256));
allDist{7,2}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v256, tempVGG16_norm_v256_split3, hof_norm_v256, hog_norm_v256, mbhx_norm_v256, mbhy_norm_v256));
allDist{7,3}=temp * temp';

temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v512, tempVGG16_norm_v512_split1, hof_norm_v512, hog_norm_v512, mbhx_norm_v512, mbhy_norm_v512));
allDist{8,1}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v512, tempVGG16_norm_v512_split2, hof_norm_v512, hog_norm_v512, mbhx_norm_v512, mbhy_norm_v512));
allDist{8,2}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_v512, tempVGG16_norm_v512_split3, hof_norm_v512, hog_norm_v512, mbhx_norm_v512, mbhy_norm_v512));
allDist{8,3}=temp * temp';

temp=NormalizeRowsUnit(cat(2,spVGG19_norm_spV32, tempVGG16_norm_spV32_split1,hof_norm_spV32, hog_norm_spV32, mbhx_norm_spV32, mbhy_norm_spV32));
allDist{9,1}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_spV32, tempVGG16_norm_spV32_split2,hof_norm_spV32, hog_norm_spV32, mbhx_norm_spV32, mbhy_norm_spV32));
allDist{9,2}=temp * temp';
temp=NormalizeRowsUnit(cat(2,spVGG19_norm_spV32, tempVGG16_norm_spV32_split3,hof_norm_spV32, hog_norm_spV32, mbhx_norm_spV32, mbhy_norm_spV32));
allDist{9,3}=temp * temp';

clear temp

all_clfsOut=cell(nEncoding,3);
all_accuracy=cell(nEncoding,3);

mean_all_accuracy=cell(nEncoding,1);

cRange = 100;
nReps = 1;
nFolds = 3;

parpool(9);
parfor k=1:nEncoding
    k
    for i=1:3
        trainI=splits(:,i)==1;
        testI=~trainI;

        trainDist = allDist{k,i}(trainI, trainI);
        testDist = allDist{k,i}(testI, trainI);
        trainLabs = labs(trainI,:);
        testLabs = labs(testI, :);

        [~, all_clfsOut{k,i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        all_accuracy{k,i} = ClassificationAccuracy(clfsOut, testLabs);
        fprintf('accuracy(%d,%d): %.3f\n', k,i, mean(all_accuracy{k,i}));

        %all_clfsOut{k,i}=clfsOut;
        %all_accuracy{k,i}=accuracy;
    end
end
delete(gcp('nocreate'))


finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    mean_all_accuracy{j}=(all_accuracy{j,1} + all_accuracy{j,2} + all_accuracy{j,3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end

