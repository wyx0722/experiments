
% tS=tic;
% videoDesc=cell(4, 1);
% 
% videoDesc{1}='/home/ionut/Data/results_desc_IDT/videoRep/FEVidHOG_IDTIDTfeatureHOGMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__VLAD256_doClassification_IDT_HOG_halfPCA_ROOTSIFT__.mat';
% videoDesc{2}='/home/ionut/Data/results_desc_IDT/videoRep/FEVidHOF_IDTIDTfeatureHOFMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim54__VLAD256_doClassification_IDT_HOF_halfPCA_ROOTSIFT__.mat';
% videoDesc{3}='/home/ionut/Data/results_desc_IDT/videoRep/FEVidMBHx_IDTIDTfeatureMBHxMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__VLAD256_doClassification_IDT_MBHx_halfPCA_ROOTSIFT__.mat';
% videoDesc{4}='/home/ionut/Data/results_desc_IDT/videoRep/FEVidMBHy_IDTIDTfeatureMBHyMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__VLAD256_doClassification_IDT_MBHy_halfPCA_ROOTSIFT__.mat';
% 
% clusters=256;
% 
% comb_vladVectors1=[];
% comb_vladVectors2=[];
% comb_vladVectors3=[];
% 
% for d=1:length(videoDesc)
%     
%     load(videoDesc{d});
%     
%     intraN_vladVectors1=intranormalizationFeatures(vladVectors1, size(vladVectors1, 2)/clusters);
%     intraN_vladVectors2=intranormalizationFeatures(vladVectors2, size(vladVectors1, 2)/clusters);
%     intraN_vladVectors3=intranormalizationFeatures(vladVectors3, size(vladVectors1, 2)/clusters);
%     
%     comb_vladVectors1=cat(2,comb_vladVectors1, NormalizeRowsUnit(PowerNormalization(intraN_vladVectors1, 0.5)) );
%     comb_vladVectors2=cat(2,comb_vladVectors2, NormalizeRowsUnit(PowerNormalization(intraN_vladVectors2, 0.5)) );
%     comb_vladVectors3=cat(2,comb_vladVectors3, NormalizeRowsUnit(PowerNormalization(intraN_vladVectors3, 0.5)) );   
% end
% tstop=toc(tS)
% 
% 
% clear intraN_vladVectors1
% clear intraN_vladVectors2
% clear intraN_vladVectors3
% 
% 
% 
% 
% [vids, labs, groups] = GetVideosPlusLabels('Full');


%% Do classification
nEncoding=3;

allDist=cell(1, nEncoding);

n_comb_vladVectors1=NormalizeRowsUnit(PowerNormalization(comb_vladVectors1, 0.5));
allDist{1}=n_comb_vladVectors1 * n_comb_vladVectors1';
clear n_comb_vladVectors1

n_comb_vladVectors2=NormalizeRowsUnit(PowerNormalization(comb_vladVectors2, 0.5));
allDist{2}=n_comb_vladVectors2 * n_comb_vladVectors2';
clear n_comb_vladVectors2

n_comb_vladVectors3=NormalizeRowsUnit(PowerNormalization(comb_vladVectors3, 0.5));
allDist{3}=n_comb_vladVectors3 * n_comb_vladVectors3';
clear n_comb_vladVectors3




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