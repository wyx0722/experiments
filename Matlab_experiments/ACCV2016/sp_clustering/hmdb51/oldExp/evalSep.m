

%% Do classification
nEncoding=3;
allDist=cell(1, nEncoding);
alpha=0.5;

d=descParam.pcaDim
clD=descParam.Clusters(1)
spClD=descParam.spClusters(3)

temp=NormalizeRowsUnit(PowerNormalization(spV64(:, clD*d+1:end), alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV64(:, clD*d+1:end-(spClD*clD)), alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV64(:, end-(spClD*clD-1):end), alpha));
allDist{3}=temp * temp';


clear temp

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;

parpool(nEncoding);
parfor k=1:nEncoding
    k
    trainI=trainTestSplit==1;
    testI=~trainI;
    
    trainDist = allDist{k}(trainI, trainI);
    testDist = allDist{k}(testI, trainI);
    trainLabs = trainTestSetlabs(trainI,:);
    testLabs = trainTestSetlabs(testI, :);
    
    [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy = ClassificationAccuracy(clfsOut, testLabs);
    fprintf('accuracy: %.3f\n', accuracy);
    
    all_clfsOut{k}=clfsOut;
    all_accuracy{k}=accuracy;
end

delete(gcp('nocreate'))