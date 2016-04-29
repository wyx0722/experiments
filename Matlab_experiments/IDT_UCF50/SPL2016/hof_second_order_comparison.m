

global DATAopts;
DATAopts = UCFInit;

[vids, labs, groups] = GetVideosPlusLabels('Full');



videoDesc=cell(4, 1);

hof='FEVidHOF_IDTIDTfeatureHOFMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim54__VLAD256_doClassification_IDT_HOF_halfPCA_ROOTSIFT__.mat';

hog='FEVidHOG_IDTIDTfeatureHOGMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__VLAD256_doClassification_IDT_HOG_halfPCA_ROOTSIFT__.mat';

mbhx='FEVidMBHx_IDTIDTfeatureMBHxMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__VLAD256_doClassification_IDT_MBHx_halfPCA_ROOTSIFT__.mat';

mbhy='FEVidMBHy_IDTIDTfeatureMBHyMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__VLAD256_doClassification_IDT_MBHy_halfPCA_ROOTSIFT__.mat';



nameD=['/home/ionut/Data/results/results_desc_IDT/videoRep/' hof ]
load(nameD);


clusters=256;


intraN_vladVectors1=intranormalizationFeatures(vladVectors1, size(vladVectors1, 2)/(clusters*1));
intraN_vladVectors2=intranormalizationFeatures(vladVectors2, size(vladVectors2, 2)/(clusters*2));
intraN_vladVectors3=intranormalizationFeatures(vladVectors3, size(vladVectors3, 2)/(clusters*3));


%% Do classification
nEncoding=8;

allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors1, 0.5));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors2, 0.5));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors3, 0.5));
allDist{3}=temp * temp';

sizeV=size(intraN_vladVectors2, 2)/2;

temp=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors2(:,1:sizeV), 0.5));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors2(:, sizeV+1:end), 0.5));
allDist{5}=temp * temp';


temp=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors3(:,1:sizeV), 0.5));
allDist{6}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors3(:,sizeV+1:2*sizeV), 0.5));
allDist{7}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors3(:,2*sizeV+1:end), 0.5));
allDist{8}=temp * temp';



parpool(9);

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

mean(mean(cat(2, accuracy{:}))')

end

delete(gcp('nocreate'))


