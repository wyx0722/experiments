global DATAopts;
DATAopts = UCFInit;

[vids, labs, groups] = GetVideosPlusLabels('Full');





%% Do classification


%fc6SpSplit1_vlad1      fc6TempSplit1_vlad1    pool4SpSplit1_vlad1    pool4TempSplit1_vlad1  pool5SpSplit1_vlad1    pool5TempSplit1_vlad1



nEncoding=7;
allDist=cell(1, nEncoding);

n_vlad1=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(pool4SpSplit1_vlad1, 0.5)), ...
         NormalizeRowsUnit(PowerNormalization(pool5SpSplit1_vlad1, 0.5))) );
allDist{1}=n_vlad1 * n_vlad1';
clear n_vlad1

n_vlad2=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(pool4SpSplit1_vlad1, 0.5)), ...
         NormalizeRowsUnit(PowerNormalization(pool5SpSplit1_vlad1, 0.5)), ...
        NormalizeRowsUnit(PowerNormalization(fc6SpSplit1_vlad1, 0.5))) );
allDist{2}=n_vlad2 * n_vlad2';
clear n_vlad2


n_vlad3=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(pool4TempSplit1_vlad1, 0.5)), ...
         NormalizeRowsUnit(PowerNormalization(pool5TempSplit1_vlad1, 0.5))) );
allDist{3}=n_vlad3 * n_vlad3';
clear n_vlad3


n_vlad4=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(pool4TempSplit1_vlad1, 0.5)), ...
         NormalizeRowsUnit(PowerNormalization(pool5TempSplit1_vlad1, 0.5)), ...
        NormalizeRowsUnit(PowerNormalization(fc6TempSplit1_vlad1, 0.5))) );
allDist{4}=n_vlad4 * n_vlad4';
clear n_vlad4


n_vlad5=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(pool5SpSplit1_vlad1, 0.5)), ...
         NormalizeRowsUnit(PowerNormalization(pool5TempSplit1_vlad1, 0.5))) );
allDist{5}=n_vlad5 * n_vlad5';
clear n_vlad5

n_vlad6=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(pool5SpSplit1_vlad1, 0.5)), ...
        NormalizeRowsUnit(PowerNormalization(pool5TempSplit1_vlad1, 0.5)), ...
        NormalizeRowsUnit(PowerNormalization(pool4SpSplit1_vlad1, 0.5)), ...
        NormalizeRowsUnit(PowerNormalization(pool4TempSplit1_vlad1, 0.5))));
allDist{6}=n_vlad6 * n_vlad6';
clear n_vlad6


n_vlad7=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(pool5SpSplit1_vlad1, 0.5)), ...
        NormalizeRowsUnit(PowerNormalization(pool5TempSplit1_vlad1, 0.5)), ...
        NormalizeRowsUnit(PowerNormalization(pool4SpSplit1_vlad1, 0.5)), ...
        NormalizeRowsUnit(PowerNormalization(pool4TempSplit1_vlad1, 0.5)), ... 
        NormalizeRowsUnit(PowerNormalization(fc6SpSplit1_vlad1, 0.5)), ...
        NormalizeRowsUnit(PowerNormalization(fc6TempSplit1_vlad1, 0.5))));
allDist{7}=n_vlad7 * n_vlad7';
clear n_vlad7




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