bdir='/home/ionut/Data/results/ICPR2016_rezults/videoRep/encoding/'

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit1netSpVGG19numClusters256pcaDim0_vladNoMean_.mat']);
split1_spVGG19=vladNoMean;

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit2netSpVGG19numClusters256pcaDim0_vladNoMean_.mat']);
split2_spVGG19=vladNoMean;

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit3netSpVGG19numClusters256pcaDim0_vladNoMean_.mat']);
split3_spVGG19=vladNoMean;




load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit1netTempVGG16Split1numClusters256pcaDim0_vladNoMean_.mat']);
split1_TempVGG16Split1=vladNoMean;

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit2netTempVGG16Split2numClusters256pcaDim0_vladNoMean_.mat']);
split2_TempVGG16Split2=vladNoMean;

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit3netTempVGG16Split2numClusters256pcaDim0_vladNoMean_.mat']);
split3_TempVGG16Split3=vladNoMean;


%% Do classification

nEncoding=3;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(split1_spVGG19, 0.5)), ...
                        NormalizeRowsUnit(PowerNormalization(split1_TempVGG16Split1, 0.5))) );
allDist{1}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(split2_spVGG19, 0.5)), ...
                        NormalizeRowsUnit(PowerNormalization(split2_TempVGG16Split2, 0.5))) );
allDist{2}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(split3_spVGG19, 0.5)), ...
                        NormalizeRowsUnit(PowerNormalization(split3_TempVGG16Split3, 0.5))) );
allDist{3}=temp * temp';

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
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy = ClassificationAccuracy(clfsOut, testLabs);
    fprintf('accuracy: %.3f\n', mean(accuracy));
    
    all_clfsOut{k}=clfsOut;
    all_accuracy{k}=accuracy;
    k
    mean(accuracy)
end

delete(gcp('nocreate'))