global DATAopts;
DATAopts = UCF101Init;
[allVids, labs, splits] = GetVideosPlusLabels('Challenge');
trainTestSplit=splits(:, 2);



bdir='/home/ionut/Data/results/ICPR2016_rezults/videoRep/'

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationROOTSIFTSplit2netSpSplit2VGG16numClusters256pcaDim256_VLAD_.mat']);
vlad1_SpSplit2VGG16=vlad1;
vlad2_SpSplit2VGG16=vlad2;
vlad3_SpSplit2VGG16=vlad3;

load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationROOTSIFTSplit2netSpVGG19numClusters256pcaDim256_VLAD_.mat']);
vlad1_SpSplit2VGG19=vlad1;
vlad2_SpSplit2VGG19=vlad2;
vlad3_SpSplit2VGG19=vlad3;


load([bdir 'FEVid_deepFeaturesDatasetUCF101Layerpool5MediaTypeDeepFNormalisationROOTSIFTSplit2nettempSplit2VGG16numClusters256pcaDim256_VLAD_.mat']);
vlad1_tempSplit2VGG16=vlad1;
vlad2_tempSplit2VGG16=vlad2;
vlad3_tempSplit2VGG16=vlad3;





%% Do classification

nEncoding=6;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(vlad1_SpSplit2VGG16, 0.5)), ...
                        NormalizeRowsUnit(PowerNormalization(vlad1_tempSplit2VGG16, 0.5))) );
allDist{1}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(vlad1_SpSplit2VGG19, 0.5)), ...
                        NormalizeRowsUnit(PowerNormalization(vlad1_tempSplit2VGG16, 0.5))) );
allDist{2}=temp * temp';



temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(vlad2_SpSplit2VGG16, 0.5)), ...
                        NormalizeRowsUnit(PowerNormalization(vlad2_tempSplit2VGG16, 0.5))) );
allDist{3}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(vlad2_SpSplit2VGG19, 0.5)), ...
                        NormalizeRowsUnit(PowerNormalization(vlad2_tempSplit2VGG16, 0.5))) );
allDist{4}=temp * temp';


temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(vlad3_SpSplit2VGG16, 0.5)), ...
                        NormalizeRowsUnit(PowerNormalization(vlad3_tempSplit2VGG16, 0.5))) );
allDist{5}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(vlad3_SpSplit2VGG19, 0.5)), ...
                        NormalizeRowsUnit(PowerNormalization(vlad3_tempSplit2VGG16, 0.5))) );
allDist{6}=temp * temp';



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