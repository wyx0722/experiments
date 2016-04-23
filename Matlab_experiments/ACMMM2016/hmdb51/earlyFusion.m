
global DATAopts;
DATAopts = HMDB51Init;


[allVids, labs, splits] = GetVideosPlusLabels();


bdir='/home/ionut/Data/results/ICPR2016_rezults/videoRep/encoding/'


nameF=[bdir 'FEVid_deepFeaturesDatasetHMBD51Split1Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19numClusters256pcaDim0_vladNoMean_maxEncode_fisherVectors_.mat']
load(nameF);
split1_spVGG19_vladNoMean=vladNoMean;
split1_spVGG19_maxEncode=maxEncode;
split1_spVGG19_fisherVectors=fisherVectors;
clear vladNoMean maxEncode fisherVectors


nameF=[bdir 'FEVid_deepFeaturesDatasetHMBD51Split2Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19numClusters256pcaDim0_vladNoMean_maxEncode_fisherVectors_.mat']
load(nameF);
split2_spVGG19_vladNoMean=vladNoMean;
split2_spVGG19_maxEncode=maxEncode;
split2_spVGG19_fisherVectors=fisherVectors;
clear vladNoMean maxEncode fisherVectors


nameF=[bdir 'FEVid_deepFeaturesDatasetHMBD51Split3Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19numClusters256pcaDim0_vladNoMean_maxEncode_fisherVectors_.mat']
load(nameF);
split3_spVGG19_vladNoMean=vladNoMean;
split3_spVGG19_maxEncode=maxEncode;
split3_spVGG19_fisherVectors=fisherVectors;
clear vladNoMean maxEncode fisherVectors


nameF=[bdir 'FEVid_deepFeaturesDatasetHMBD51Split1Layerpool5MediaTypeDeepFNormalisationNonenettempSplit1VGG16numClusters256pcaDim0_vladNoMean_maxEncode_fisherVectors_.mat']
load(nameF);
split1_tempVGG16_vladNoMean=vladNoMean;
split1_tempVGG16_maxEncode=maxEncode;
split1_tempVGG16_fisherVectors=fisherVectors;
clear vladNoMean maxEncode fisherVectors

nameF=[bdir 'FEVid_deepFeaturesDatasetHMBD51Split2Layerpool5MediaTypeDeepFNormalisationNonenettempSplit1VGG16numClusters256pcaDim0_vladNoMean_maxEncode_fisherVectors_.mat']
load(nameF);
split2_tempVGG16_vladNoMean=vladNoMean;
split2_tempVGG16_maxEncode=maxEncode;
split2_tempVGG16_fisherVectors=fisherVectors;
clear vladNoMean maxEncode fisherVectors

nameF=[bdir 'FEVid_deepFeaturesDatasetHMBD51Split3Layerpool5MediaTypeDeepFNormalisationNonenettempSplit1VGG16numClusters256pcaDim0_vladNoMean_maxEncode_fisherVectors_.mat']
load(nameF);
split3_tempVGG16_vladNoMean=vladNoMean;
split3_tempVGG16_maxEncode=maxEncode;
split3_tempVGG16_fisherVectors=fisherVectors;
clear vladNoMean maxEncode fisherVectors


nEncoding=9;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(split1_spVGG19_vladNoMean), NormalizeRowsUnit(split1_tempVGG16_vladNoMean)) );
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(split2_spVGG19_vladNoMean), NormalizeRowsUnit(split2_tempVGG16_vladNoMean)));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(split3_spVGG19_vladNoMean), NormalizeRowsUnit(split3_tempVGG16_vladNoMean)));
allDist{3}=temp * temp';


temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(split1_spVGG19_maxEncode), NormalizeRowsUnit(split1_tempVGG16_maxEncode)) );
allDist{4}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(split2_spVGG19_maxEncode), NormalizeRowsUnit(split2_tempVGG16_maxEncode)) );
allDist{5}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(split3_spVGG19_maxEncode), NormalizeRowsUnit(split3_tempVGG16_maxEncode)) );
allDist{6}=temp * temp';


temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(split1_spVGG19_fisherVectors), NormalizeRowsUnit(split1_tempVGG16_fisherVectors)) );
allDist{7}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(split2_spVGG19_fisherVectors), NormalizeRowsUnit(split2_tempVGG16_fisherVectors)) );
allDist{8}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(split3_spVGG19_fisherVectors), NormalizeRowsUnit(split3_tempVGG16_fisherVectors)) );
allDist{9}=temp * temp';


all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;

sp=[1 2 3 1 2 3 1 2 3 ]

parpool(nEncoding);
parfor k=1:nEncoding
    k
    trainTestSplit=splits((splits(:, sp(k))==1 | splits(:, sp(k))==2), sp(k));
    trainI=trainTestSplit==1;
    testI=~trainI;
    
    trainTestSetlabs=labs((splits(:, sp(k))==1 | splits(:, sp(k))==2), :);
    
    trainDist = allDist{k}(trainI, trainI);
    testDist = allDist{k}(testI, trainI);
    trainLabs = trainTestSetlabs(trainI,:);
    testLabs = trainTestSetlabs(testI, :);
    
    [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy = ClassificationAccuracy(clfsOut, testLabs);
    fprintf('accuracy: %.3f\n', accuracy);
    
    k
    mean(accuracy)
    
    all_clfsOut{k}=clfsOut;
    all_accuracy{k}=accuracy;
end

delete(gcp('nocreate'))