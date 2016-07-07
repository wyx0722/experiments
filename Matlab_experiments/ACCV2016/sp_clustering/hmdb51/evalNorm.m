

%% Do classification
nEncoding=4;
allDist=cell(1, nEncoding);
alpha=0.5;

d=descParam.pcaDim
clD=descParam.Clusters(1)
spClD=descParam.spClusters(3)


spV64_intraL2=zeros(size(spV64));
spV64_intraL2(:, 1:end-(spClD*clD))=intranormalizationFeatures( spV64(:, 1:end-(spClD*clD)), d );
spV64_intraL2(:, end-(spClD*clD-1):end)=intranormalizationFeatures(spV64(:, end-(spClD*clD-1):end),clD );

spV64_intraPNL2=zeros(size(spV64));
spV64_intraPNL2(:, 1:end-(spClD*clD))=intranormalizationFeatures_L2_PN( spV64(:, 1:end-(spClD*clD)), d, alpha );
spV64_intraPNL2(:, end-(spClD*clD-1):end)=intranormalizationFeatures_L2_PN(spV64(:, end-(spClD*clD-1):end),clD, alpha );


spV64_compL2=zeros(size(spV64));
spV64_compL2(:, 1:clD*d)=NormalizeRowsUnit(spV64(:, 1:clD*d));
spV64_compL2(:, clD*d+1:end-(spClD*clD))=NormalizeRowsUnit(spV64(:, clD*d+1:end-(spClD*clD)));
spV64_compL2(:, end-(spClD*clD-1):end)=NormalizeRowsUnit(spV64(:, end-(spClD*clD-1):end));

spV64_compPNL2=zeros(size(spV64));
spV64_compPNL2(:, 1:clD*d)=NormalizeRowsUnit(PowerNormalization(spV64(:, 1:clD*d), alpha));
spV64_compPNL2(:, clD*d+1:end-(spClD*clD))=NormalizeRowsUnit(PowerNormalization(spV64(:, clD*d+1:end-(spClD*clD)), alpha));
spV64_compPNL2(:, end-(spClD*clD-1):end)=NormalizeRowsUnit(PowerNormalization(spV64(:, end-(spClD*clD-1):end), alpha));



temp=NormalizeRowsUnit(spV64_intraL2);
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(spV64_intraPNL2);
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(spV64_compL2);
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(spV64_compPNL2);
allDist{4}=temp * temp';

clear temp

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
    trainLabs = trainTestSetlabs(trainI,:);
    testLabs = trainTestSetlabs(testI, :);
    
    [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy = ClassificationAccuracy(clfsOut, testLabs);
    fprintf('accuracy: %.3f\n', accuracy);
    
    all_clfsOut{k}=clfsOut;
    all_accuracy{k}=accuracy;
end

delete(gcp('nocreate'))