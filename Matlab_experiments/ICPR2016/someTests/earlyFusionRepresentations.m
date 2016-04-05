
nEncoding=4;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(maxEncode, 0.5)), ...
     NormalizeRowsUnit(PowerNormalization(vladNoMean, 0.5)) ));
allDist{1}=temp * temp';


temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(maxEncode, 0.5)), ...
     NormalizeRowsUnit(PowerNormalization(fisherVectors, 0.5)) ));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(vladNoMean, 0.5)), ...
     NormalizeRowsUnit(PowerNormalization(fisherVectors, 0.5)) ));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(maxEncode, 0.5)), ...
     NormalizeRowsUnit(PowerNormalization(vladNoMean, 0.5)), ...
     NormalizeRowsUnit(PowerNormalization(fisherVectors, 0.5)) ));
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