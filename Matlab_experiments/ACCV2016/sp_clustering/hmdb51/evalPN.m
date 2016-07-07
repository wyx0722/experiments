

%% Do classification
nEncoding=8;
allDist=cell(1, nEncoding);
alpha=[0.1:0.1:0.4];


for i=1:length(alpha)
temp=NormalizeRowsUnit(PowerNormalization(v256, alpha(i)));
allDist{i}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV64, alpha(i)));
allDist{4+i}=temp * temp';

end

clear temp

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;

parpool(2);
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
mean(all_accuracy{1})