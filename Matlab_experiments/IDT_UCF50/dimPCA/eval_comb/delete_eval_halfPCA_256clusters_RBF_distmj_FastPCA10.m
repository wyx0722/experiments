
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

pca_comb_vladVectors1=FastPCA(comb_vladVectors1,ceil(size(comb_vladVectors1, 2)/10));
r_comb_vladVectors1=comb_vladVectors1 * pca_comb_vladVectors1.data.rot;
allDist{1}=distmj(r_comb_vladVectors1, r_comb_vladVectors1);



pca_comb_vladVectors2=FastPCA(comb_vladVectors2,ceil(size(comb_vladVectors2, 2)/10));
r_comb_vladVectors2=comb_vladVectors2 * pca_comb_vladVectors2.data.rot;
allDist{2}=distmj(r_comb_vladVectors2, r_comb_vladVectors2);

pca_comb_vladVectors3=FastPCA(comb_vladVectors3,ceil(size(comb_vladVectors3, 2)/10));
r_comb_vladVectors3=comb_vladVectors3 * pca_comb_vladVectors3.data.rot;
allDist{3}=distmj(r_comb_vladVectors3 , r_comb_vladVectors3);





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
    
    [~, clfsOut{i}] = SvmPKExpOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

all_clfsOut{k}=clfsOut;
all_accuracy{k}=accuracy;

k
perGroupAccuracy = mean(cat(2, accuracy{:}))'

end

delete(gcp('nocreate'))