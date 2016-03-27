nEncoding=2;
allDist=cell(1, nEncoding);

n_vlad1=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(vlad1, 0.5)), NormalizeRowsUnit(PowerNormalization(spVLAD1, 0.5))));
allDist{1}=n_vlad1 * n_vlad1';
clear n_vlad1

n_vlad2=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(vlad2, 0.5)), NormalizeRowsUnit(PowerNormalization(spVLAD2, 0.5))));
allDist{2}=n_vlad2 * n_vlad2';
clear n_vlad2


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
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy = ClassificationAccuracy(clfsOut, testLabs);
    fprintf('accuracy: %.3f\n', accuracy);
    
    all_clfsOut{k}=clfsOut;
    all_accuracy{k}=accuracy;
end

delete(gcp('nocreate'))

mean(all_accuracy{1})
mean(all_accuracy{2})