
nEncoding=8;


%% Do classification

allDist=cell(1, nEncoding);

impr1=NormalizeRowsUnit(SquareRootAbs(vladVectors2(:, 1:dimVlad)));
allDist{1}=impr1 * impr1';
clear impr1

impr2=NormalizeRowsUnit(SquareRootAbs(vladVectors2(:, dimVlad+1:end)));
allDist{2}=impr2 * impr2';
clear impr2

boost1=NormalizeRowsUnit(SquareRootAbs(vladVectors3(:, 1:dimVlad)));
allDist{3}=boost1 * boost1';
clear boost1

boost2=NormalizeRowsUnit(SquareRootAbs(vladVectors3(:, dimVlad+1:2*dimVlad)));
allDist{4}=boost2 * boost2';
clear boost2

boost3=NormalizeRowsUnit(SquareRootAbs(vladVectors3(:, 2*dimVlad+1:end)));
allDist{5}=boost3 * boost3';
clear boost3


n_vladVectors1=NormalizeRowsUnit(SquareRootAbs(vladVectors1));
allDist{6}=n_vladVectors1 * n_vladVectors1';
clear n_vladVectors1

n_vladVectors2=NormalizeRowsUnit(SquareRootAbs(vladVectors2));
allDist{7}=n_vladVectors2 * n_vladVectors2';
clear n_vladVectors2

n_vladVectors3=NormalizeRowsUnit(SquareRootAbs(vladVectors3));
allDist{8}=n_vladVectors3 * n_vladVectors3';
clear n_vladVectors3



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



% saveName = [DATAopts.resultsPath DescParam2Name(descParam) 'VLAD_1_mean_512C.mat'];
% save(saveName, '-v7.3', 'descParam', 'clfsOut', 'accuracy', 'vectNrFrames', 'tFinalWorldA');

% saveName2 = [DATAopts.featurePath DescParam2Name(descParam) 'VLADE_1_512C.mat'];
% save(saveName2, '-v7.3', 'descParam', 'vladVectors', 'groups', 'labs');