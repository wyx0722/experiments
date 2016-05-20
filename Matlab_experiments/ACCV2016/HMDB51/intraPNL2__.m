initDim=128;

intraPNL2_vladNoMean = intranormalizationFeatures_L2_PN( vladNoMean, initDim, 0.5 );
intraPNL2_maxPool = intranormalizationFeatures_L2_PN( maxPool, initDim, 0.5 );
intraPNL2_multiVLAD = intranormalizationFeatures_L2_PN( multiVLAD, initDim, 0.5 );
intraPNL2_multiMaxPool = intranormalizationFeatures_L2_PN( multiMaxPool, initDim, 0.5 );

nPN_multiVLAD=intranormalizationFeatures_L2_PN( multiVLAD, 128*32, 0.5 );
nPN_multiMaxPool = intranormalizationFeatures_L2_PN( multiMaxPool, 128*32, 0.5);


%% Do classification
alpha=0.5;

nEncoding=6;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(PowerNormalization(intraPNL2_vladNoMean, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraPNL2_maxPool, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraPNL2_multiVLAD, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraPNL2_multiMaxPool, alpha));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(nPN_multiVLAD, alpha));
allDist{5}=temp * temp';


temp=NormalizeRowsUnit(PowerNormalization(nPN_multiMaxPool, alpha));
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
    trainLabs = trainTestSetlabs(trainI,:);
    testLabs = trainTestSetlabs(testI, :);
    
    [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy = ClassificationAccuracy(clfsOut, testLabs);
    fprintf('accuracy: %.3f\n', accuracy);
    
    all_clfsOut{k}=clfsOut;
    all_accuracy{k}=accuracy;
end

delete(gcp('nocreate'))

acc1=mean(all_accuracy{1})
acc2=mean(all_accuracy{2})
acc3=mean(all_accuracy{3})
acc4=mean(all_accuracy{4})
acc5=mean(all_accuracy{5})
acc6=mean(all_accuracy{6})


% fileName=sprintf('/home/ionut/experiments/Matlab_experiments/ACMMM2016/results/results_HMDB51_Features%s_Layer%s_Network%s_PCAdim%d_clusters%d_norm%s__VLAD.txt', descParam.MediaType,descParam.Layer, descParam.net,descParam.pcaDim, descParam.numClusters, descParam.Normalisation); 
% fileID=fopen(fileName, 'a');
% fprintf(fileID, 'Dataset and Split: %s --> vladNoMean acc= %.4f   maxEncode acc= %.4f  fisherVectors acc= %.4f \r\n', descParam.Dataset, acc1,acc2, acc3 );
% fclose(fileID);
% 
% 
% saveName = ['/home/ionut/Data/results/ICPR2016_rezults/' 'clfsOut/' 'encoding/'  DescParam2Name(descParam) '_vladNoMean_maxEncode_fisherVectors_.mat'];
% save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');
% 
%  saveName2 = ['/home/ionut/Data/results/ICPR2016_rezults/' 'videoRep/' 'encoding/' DescParam2Name(descParam) '_vladNoMean_maxEncode_fisherVectors_.mat'];
%  save(saveName2, '-v7.3', 'vladNoMean', 'maxEncode', 'fisherVectors');
%  
% 
%  
% 
% end