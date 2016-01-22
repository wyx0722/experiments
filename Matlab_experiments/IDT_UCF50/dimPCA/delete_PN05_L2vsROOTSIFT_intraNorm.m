
intraN_vladVectors1=intranormalizationFeatures(vladVectors1, size(vocabulary, 2));
intraN_vladVectors2=intranormalizationFeatures(vladVectors2, size(vocabulary, 2));
intraN_vladVectors3=intranormalizationFeatures(vladVectors3, size(vocabulary, 2));


%% Do classification

nEncoding=3;

allDist=cell(1, nEncoding);

n_vladVectors1=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors1, 0.5));
allDist{1}=n_vladVectors1 * n_vladVectors1';
clear n_vladVectors1

n_vladVectors2=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors2, 0.5));
allDist{2}=n_vladVectors2 * n_vladVectors2';
clear n_vladVectors2

n_vladVectors3=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors3, 0.5));
allDist{3}=n_vladVectors3 * n_vladVectors3';
clear n_vladVectors3

oldPN05_all_clfsOut=all_clfsOut;
oldPN05_all_accuracy=all_accuracy;

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;

parpool(5);

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

end

delete(gcp('nocreate'))


saveName = ['/home/ionut/Data/results_desc_IDT/' 'clfsOut/' DescParam2Name(descParam) '__VLAD256_doClassification_IDT____max_no_PCA_PN05_intraN__.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = ['/home/ionut/Data/results_desc_IDT/' 'videoRep/' DescParam2Name(descParam) '__VLAD256_doClassification_IDT____max_no_PCA_PN05_intraN__.mat'];
 save(saveName2, '-v7.3', 'vladVectors1', 'vladVectors2', 'vladVectors3');


