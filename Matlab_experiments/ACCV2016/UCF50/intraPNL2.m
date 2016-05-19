initDim=256;

intraPNL2_vladNoMean = intranormalizationFeatures_L2_PN( vladNoMean, initDim, 0.5 );
intraPNL2_vladMean = intranormalizationFeatures_L2_PN( vladMean, initDim, 0.5 );
intraPNL2_stdDiff = intranormalizationFeatures_L2_PN( stdDiff, initDim, 0.5 );
intraPNL2_stdDiffMean = intranormalizationFeatures_L2_PN( stdDiffMean, initDim, 0.5 );


%% Do classification
alpha=0.5;

nEncoding=4;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(PowerNormalization(intraPNL2_vladNoMean, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraPNL2_vladMean, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraPNL2_stdDiff, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraPNL2_stdDiffMean, alpha));
allDist{4}=temp * temp';


clear temp


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

mean(perGroupAccuracy)

end

delete(gcp('nocreate'))


% saveName = ['/home/ionut/Data/results/ICPR2016_rezults/' 'clfsOut/' 'encoding/'  DescParam2Name(descParam) '_vladNoMean_maxEncode_fisherVectors_.mat'];
% save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');
% 
%  saveName2 = ['/home/ionut/Data/results/ICPR2016_rezults/' 'videoRep/' 'encoding/' DescParam2Name(descParam) '_vladNoMean_maxEncode_fisherVectors_.mat'];
%  save(saveName2, '-v7.3', 'vladNoMean', 'maxEncode', 'fisherVectors');
%  
 

