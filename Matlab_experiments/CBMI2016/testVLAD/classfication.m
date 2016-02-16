
[vids, labs, groups] = GetVideosPlusLabels('Full');

nEncoding=1;



load(path);

n_VLADAll=NormalizeRowsUnit(PowerNormalization(VLADAll, 0.5));
allDist{1}=n_VLADAll * n_VLADAll';
clear n_VLADAll


all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);


parpool(12);

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
    
    [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs);
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

all_clfsOut{k}=clfsOut;
all_accuracy{k}=accuracy;

k
perGroupAccuracy = mean(cat(2, accuracy{:}))'

end



acc=mean(mean(cat(2, all_accuracy{1}{:})))


delete(gcp('nocreate'))


