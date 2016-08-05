
nEncoding=2;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(repVocabSim);
allDist{1}=temp * temp';

in_repVocabSim=1./repVocabSim; %!!!!!!!!!generates Inf!!!!!!!
in_repVocabSim(isinf(in_repVocabSim))=0;
temp=NormalizeRowsUnit(in_repVocabSim); 
allDist{2}=temp * temp';

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

acc1=mean(all_accuracy{1})
acc2=mean(all_accuracy{2})
