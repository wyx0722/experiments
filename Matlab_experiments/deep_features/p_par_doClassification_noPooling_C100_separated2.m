
nEncoding=3;


%% Do classification

allDist=cell(1, nEncoding);

impr1=NormalizeRowsUnit(SquareRootAbs(vladVectors2(:, 1:dimVlad)));
impr2=NormalizeRowsUnit(SquareRootAbs(vladVectors2(:, dimVlad+1:end)));

impr_1_2=NormalizeRowsUnit(SquareRootAbs(cat(2,impr1, impr2 )));
clear impr1

allDist{1}=impr_1_2 * impr_1_2';
clear impr_1_2;


boost1=NormalizeRowsUnit(SquareRootAbs(vladVectors3(:, 1:dimVlad)));
boost2=NormalizeRowsUnit(SquareRootAbs(vladVectors3(:, dimVlad+1:2*dimVlad)));
boost3=NormalizeRowsUnit(SquareRootAbs(vladVectors3(:, 2*dimVlad+1:end)));

boost_1_2_3=NormalizeRowsUnit(SquareRootAbs(cat(2,boost1, boost2, boost3 )));
allDist{2}=boost_1_2_3 * boost_1_2_3';
clear boost_1_2_3
clear boost2
clear boost3

boost1_impr2=NormalizeRowsUnit(SquareRootAbs(cat(2,boost1, impr2 )));

allDist{3}=boost1_impr2 * boost1_impr2';
clear boost1_impr2
clear boost1
clear impr2


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