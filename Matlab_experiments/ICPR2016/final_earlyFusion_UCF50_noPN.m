global DATAopts;
DATAopts = UCFInit;

[vids, labs, groups] = GetVideosPlusLabels('Full');





%% Do classification
bdir='/home/ionut/Data/results/ICPR2016_rezults/videoRep/encoding/'


nameF=[bdir 'FEVid_deepFeaturesLayerpool5MediaTypeDeepFNormalisationNonenetSpVGG19numClusters256pcaDim0_vladNoMean_.mat']
load(nameF);
spVGG19_vladNoMean=vladNoMean;
clear vladNoMean

nameF=[bdir 'FEVid_deepFeaturesLayerpool5MediaTypeDeepFNormalisationNonenetSpVGG19numClusters256pcaDim0_maxEncode_.mat']
load(nameF);
spVGG19_maxEncode=maxEncode;
clear maxEncode

nameF=[bdir 'FEVid_deepFeaturesLayerpool5MediaTypeDeepFNormalisationNonenetSpVGG19numClusters256pcaDim0_fisherVectors_.mat']
load(nameF);
spVGG19_fisherVectors=fisherVectors;
clear fisherVectors



nameF=[bdir 'FEVid_deepFeaturesLayerpool5MediaTypeDeepFNormalisationNonenetTempVGG16numClusters256pcaDim0_vladNoMean_.mat']
load(nameF);
tempVGG16_vladNoMean=vladNoMean;
clear vladNoMean

nameF=[bdir 'FEVid_deepFeaturesLayerpool5MediaTypeDeepFNormalisationNonenetTempVGG16numClusters256pcaDim0_maxEncode_.mat']
load(nameF);
tempVGG16_maxEncode=maxEncode;
clear maxEncode

nameF=[bdir 'FEVid_deepFeaturesLayerpool5MediaTypeDeepFNormalisationNonenetTempVGG16numClusters256pcaDim0_fisherVectors_.mat']
load(nameF);
tempVGG16_fisherVectors=fisherVectors;
clear fisherVectors


nEncoding=9;
allDist=cell(1, nEncoding);


n_spVGG19_vladNoMean=NormalizeRowsUnit(spVGG19_vladNoMean);
allDist{1}=n_spVGG19_vladNoMean * n_spVGG19_vladNoMean';


n_spVGG19_maxEncode=NormalizeRowsUnit(spVGG19_maxEncode);
allDist{2}=n_spVGG19_maxEncode * n_spVGG19_maxEncode';


n_spVGG19_fisherVectors=NormalizeRowsUnit(spVGG19_fisherVectors);
allDist{3}=n_spVGG19_fisherVectors * n_spVGG19_fisherVectors';



n_tempVGG16_vladNoMean=NormalizeRowsUnit(tempVGG16_vladNoMean);
allDist{4}=n_tempVGG16_vladNoMean * n_tempVGG16_vladNoMean';


n_tempVGG16_maxEncode=NormalizeRowsUnit(tempVGG16_maxEncode);
allDist{5}=n_tempVGG16_maxEncode * n_tempVGG16_maxEncode';


n_tempVGG16_fisherVectors=NormalizeRowsUnit(tempVGG16_fisherVectors);
allDist{6}=n_tempVGG16_fisherVectors * n_tempVGG16_fisherVectors';



temp=NormalizeRowsUnit( cat(2, n_spVGG19_vladNoMean, n_tempVGG16_vladNoMean));
allDist{7}=temp * temp';


temp=NormalizeRowsUnit( cat(2, n_spVGG19_maxEncode, n_tempVGG16_maxEncode));
allDist{8}=temp * temp';

temp=NormalizeRowsUnit( cat(2, n_spVGG19_fisherVectors, n_tempVGG16_fisherVectors));
allDist{9}=temp * temp';

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