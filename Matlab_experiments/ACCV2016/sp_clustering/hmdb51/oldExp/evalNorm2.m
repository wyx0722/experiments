

%% Do classification
nEncoding=5;
allDist=cell(1, nEncoding);
alpha=0.5;




temp=NormalizeRowsUnit(PowerNormalization(spV64_intraL2, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV64_intraPNL2, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV64_compL2, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV64_compPNL2, alpha));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(spV64);
allDist{5}=temp * temp';


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
mean(all_accuracy{1})