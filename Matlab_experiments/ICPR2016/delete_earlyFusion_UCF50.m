global DATAopts;
DATAopts = UCFInit;

[vids, labs, groups] = GetVideosPlusLabels('Full');





%% Do classification


%fc6SpSplit1_vlad1      fc6TempSplit1_vlad1    pool4SpSplit1_vlad1    pool4TempSplit1_vlad1  pool5SpSplit1_vlad1    pool5TempSplit1_vlad1

% spVGG19_vlad0    spVGG19_vlad1    tempVGG16_vlad0  tempVGG16_vlad1

nEncoding=2;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(spVGG19_vlad0, 0.5)), ...
         NormalizeRowsUnit(PowerNormalization(tempVGG16_vlad0, 0.5))) );
allDist{1}=temp * temp';


temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(spVGG19_vlad1, 0.5)), ...
         NormalizeRowsUnit(PowerNormalization(tempVGG16_vlad1, 0.5))) );
allDist{2}=temp * temp';




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
perGroupAccuracy = mean(mean(cat(2, accuracy{:}))')

end

delete(gcp('nocreate'))