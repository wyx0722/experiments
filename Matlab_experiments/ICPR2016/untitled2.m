
parpool(5);

parfor i=1:max(groups)
    testI = groups == i;
    trainI = ~testI;
    trainDist = allDist{3}(trainI, trainI);
    testDist = allDist{3}(testI, trainI);
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

delete(gcp('nocreate'))

all_clfsOut{3}=clfsOut;
all_accuracy{3}=accuracy;

mean(mean(cat(2, all_accuracy{3}{:})))
