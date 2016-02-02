
tS=tic;
videoDesc=cell(4, 1);

videoDesc{1}='/home/ionut/Data/results_desc_IDT/videoRep/SpatialPyramid/FEVid_IDTIDTfeatureHOG_iTrajMediaTypeIDTNormalisationL1PNalpha0.6numClusters512pcaDim96sCol1sRow3_VLAD_DV_.mat';
videoDesc{2}='/home/ionut/Data/results_desc_IDT/videoRep/SpatialPyramid/FEVid_IDTIDTfeatureHOF_iTrajMediaTypeIDTNormalisationPNL2alpha0.1numClusters512pcaDim108sCol1sRow3_VLAD_DV_.mat';
videoDesc{3}='/home/ionut/Data/results_desc_IDT/videoRep/SpatialPyramid/FEVid_IDTIDTfeatureMBHx_iTrajMediaTypeIDTNormalisationPNL2alpha1numClusters512pcaDim80sCol1sRow3_VLAD_DV_.mat';
videoDesc{4}='/home/ionut/Data/results_desc_IDT/videoRep/SpatialPyramid/FEVid_IDTIDTfeatureMBHy_iTrajMediaTypeIDTNormalisationPNL2alpha1numClusters512pcaDim64sCol1sRow3_VLAD_DV_.mat';

clusters=256;

comb_vladVectors1=[];
%comb_vladVectors2=[];
%comb_vladVectors3=[];

for d=1:length(videoDesc)
    
    load(videoDesc{d});
    
    %intraN_vladVectors1=intranormalizationFeatures(vladVectors1, size(vladVectors1, 2)/clusters);
    intraN_vladVectors2=intranormalizationFeatures(vladVectors2, size(vladVectors1, 2)/clusters);
    %intraN_vladVectors3=intranormalizationFeatures(vladVectors3, size(vladVectors1, 2)/clusters);
    
    %comb_vladVectors1=cat(2,comb_vladVectors1, NormalizeRowsUnit(PowerNormalization(intraN_vladVectors1, 0.5)) );
    comb_vladVectors2=cat(2,comb_vladVectors2, NormalizeRowsUnit(PowerNormalization(intraN_vladVectors2, 0.5)) );
    %comb_vladVectors3=cat(2,comb_vladVectors3, NormalizeRowsUnit(PowerNormalization(intraN_vladVectors3, 0.5)) );   
end
tstop=toc(tS)


%clear intraN_vladVectors1
clear intraN_vladVectors2
%clear intraN_vladVectors3




[vids, labs, groups] = GetVideosPlusLabels('Full');


%% Do classification
nEncoding=2;

allDist=cell(1, nEncoding);

n_comb_vladVectors2=NormalizeRowsUnit(comb_vladVectors2);
allDist{1}=n_comb_vladVectors2 * n_comb_vladVectors2';
clear n_comb_vladVectors2

n_comb_vladVectors2_1=NormalizeRowsUnit(comb_vladVectors2(:, 1:size(comb_vladVectors2, 2)/4));
allDist{2}=n_comb_vladVectors2_1 * n_comb_vladVectors2_1';
clear n_comb_vladVectors2_1


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

mean(mean(cat(2, accuracy{:}))')

end

delete(gcp('nocreate'))