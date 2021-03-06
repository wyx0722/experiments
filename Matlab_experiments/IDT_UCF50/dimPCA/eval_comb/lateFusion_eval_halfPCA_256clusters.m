
tS=tic;
videoDesc=cell(4, 1);

videoDesc{1}='/home/ionut/Data/results_desc_IDT/clfsOut/FEVidHOG_IDTIDTfeatureHOGMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__VLAD256_doClassification_IDT_HOG_halfPCA_ROOTSIFT__.mat';
videoDesc{2}='/home/ionut/Data/results_desc_IDT/clfsOut/FEVidHOF_IDTIDTfeatureHOFMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim54__VLAD256_doClassification_IDT_HOF_halfPCA_ROOTSIFT__.mat';
videoDesc{3}='/home/ionut/Data/results_desc_IDT/clfsOut/FEVidMBHx_IDTIDTfeatureMBHxMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__VLAD256_doClassification_IDT_MBHx_halfPCA_ROOTSIFT__.mat';
videoDesc{4}='/home/ionut/Data/results_desc_IDT/clfsOut/FEVidMBHy_IDTIDTfeatureMBHyMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48__VLAD256_doClassification_IDT_MBHy_halfPCA_ROOTSIFT__.mat';


comb_all_clfsOut=cell(4, 1);

for d=1:length(videoDesc)
    
    load(videoDesc{d});
    
    comb_all_clfsOut{d}=all_clfsOut;
    
end
tstop=toc(tS)



[vids, labs, groups] = GetVideosPlusLabels('Full');


%% Do classification
nEncoding=3;


all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);


for k=1:nEncoding

% 
% Leave-one-group-out cross-validation
for i=1:max(groups)
    testI = groups == i;
    trainI = ~testI;

    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    %[~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    
    clfsOut{i}=comb_all_clfsOut{1}{k}{i}+comb_all_clfsOut{2}{k}{i}+ ...
               comb_all_clfsOut{3}{k}{i}+comb_all_clfsOut{4}{k}{i};
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

all_clfsOut{k}=clfsOut;
all_accuracy{k}=accuracy;

k
perGroupAccuracy = mean(cat(2, accuracy{:}))'

end

