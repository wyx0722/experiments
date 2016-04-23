global DATAopts;
DATAopts = UCF101Init;
[allVids, labs, splits] = GetVideosPlusLabels('Challenge');


bdir='/home/ionut/Data/results/ICPR2016_rezults/videoRep/encoding/'

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit1netSpVGG19numClusters256pcaDim0_fisherVectors_.mat']);
split1_spVGG19=fisherVectors;

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit2netSpVGG19numClusters256pcaDim0_fisherVectors_.mat']);
split2_spVGG19=fisherVectors;

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit3netSpVGG19numClusters256pcaDim0_fisherVectors_.mat']);
split3_spVGG19=fisherVectors;




load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit1netTempVGG16Split1numClusters256pcaDim0_fisherVectors_.mat']);
split1_TempVGG16Split1=fisherVectors;

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit2netTempVGG16Split2numClusters256pcaDim0_fisherVectors_.mat']);
split2_TempVGG16Split2=fisherVectors;

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit3netTempVGG16Split2numClusters256pcaDim0_fisherVectors_.mat']);
split3_TempVGG16Split3=fisherVectors;


%% Do classification

nEncoding=9;
allDist=cell(1, nEncoding);


sp1=NormalizeRowsUnit(split1_spVGG19);
allDist{1}=sp1 * sp1';

sp2=NormalizeRowsUnit(split2_spVGG19);
allDist{2}=sp2 * sp2';

sp3=NormalizeRowsUnit(split3_spVGG19);
allDist{3}=sp3 * sp3';


temp1=NormalizeRowsUnit(split1_TempVGG16Split1);
allDist{4}=temp1 * temp1';

temp2=NormalizeRowsUnit(split2_TempVGG16Split2);
allDist{5}=temp2 * temp2';

temp3=NormalizeRowsUnit(split3_TempVGG16Split3);
allDist{6}=temp3 * temp3';


temp=NormalizeRowsUnit( cat(2, sp1, temp1));
allDist{7}=temp * temp';
clear sp1 temp1


temp=NormalizeRowsUnit( cat(2, sp2, temp2));
allDist{8}=temp * temp';
clear sp2 temp2

temp=NormalizeRowsUnit( cat(2, sp3, temp3));
allDist{9}=temp * temp';
clear sp3 temp3
clear temp

sp=[1 2 3 1 2 3 1 2 3];

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;



parpool(nEncoding);
parfor k=1:nEncoding
    k
    trainTestSplit=splits(:, sp(k));
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