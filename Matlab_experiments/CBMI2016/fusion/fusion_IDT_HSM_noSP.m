
hsm_fisher_noSP=fisherAll(:, 1:size(fisherAll,2)/4);


hsm_fisher_noSP01=NormalizeRowsUnit(PowerNormalization(hsm_fisher_noSP, 0.1));


nEncoding=1;
allDist=cell(1, nEncoding);



ul_all_IDT_hsm_noSP=NormalizeRowsUnit(cat(2,all_IDT01, hsm_fisher_noSP01));
allDist{1}=ul_all_IDT_hsm_noSP * ul_all_IDT_hsm_noSP';
clear ul_all_IDT_hsm_noSP


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

end


delete(gcp('nocreate'))


acc1=mean(mean(cat(2, all_accuracy{1}{:})))
acc2=mean(mean(cat(2, all_accuracy{2}{:})))
acc3=mean(mean(cat(2, all_accuracy{3}{:})))
acc4=mean(mean(cat(2, all_accuracy{4}{:})))






