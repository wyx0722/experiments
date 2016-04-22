
global DATAopts;
DATAopts = UCF101Init;
[allVids, labs, splits] = GetVideosPlusLabels('Challenge');
trainTestSplit=splits(:, 1);


bdir='/home/ionut/Data/results/ICPR2016_rezults/videoRep/encoding/'

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit1netSpVGG19numClusters256pcaDim0_maxEncode_.mat']);
maxEncode_split1_spVGG19=maxEncode;

% load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit2netSpVGG19numClusters256pcaDim0_maxEncode_.mat']);
% maxEncode_split2_spVGG19=maxEncode;
% 
% load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit3netSpVGG19numClusters256pcaDim0_maxEncode_.mat']);
% maxEncode_split3_spVGG19=maxEncode;




load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit1netTempVGG16Split1numClusters256pcaDim0_maxEncode_.mat']);
maxEncode_split1_TempVGG16Split1=maxEncode;

% load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit2netTempVGG16Split2numClusters256pcaDim0_maxEncode_.mat']);
% maxEncode_split2_TempVGG16Split2=maxEncode;
% 
% load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationNoneSplit3netTempVGG16Split2numClusters256pcaDim0_maxEncode_.mat']);
% maxEncode_split3_TempVGG16Split3=maxEncode;


%% Do classification

alpha=0:0.1:1;

nEncoding=length(alpha);
allDist=cell(1, nEncoding);

for i=1:length(alpha)
    temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(maxEncode_split1_spVGG19, alpha(i))), ...
                        NormalizeRowsUnit(PowerNormalization(maxEncode_split1_TempVGG16Split1, alpha(i)))) );
    allDist{i}=temp * temp';
end


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