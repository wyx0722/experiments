


%% Do classification
alpha=0.5;

nEncoding=9;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(PowerNormalization(vladNoMean, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(stdDiff, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(multiVLAD, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(multiStdDiff, alpha));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(cat(2,vladNoMean,stdDiff ), alpha));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(cat(2,multiVLAD, multiStdDiff ), alpha));
allDist{6}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(cat(2,vladNoMean, stdDiff, multiVLAD, multiStdDiff), alpha));
allDist{7}=temp * temp';


temp=NormalizeRowsUnit(PowerNormalization(cat(2, vladNoMean,multiVLAD ), alpha));
allDist{8}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(cat(2, stdDiff,multiStdDiff ), alpha));
allDist{9}=temp * temp';


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
acc7=mean(all_accuracy{7})
acc8=mean(all_accuracy{8})
acc9=mean(all_accuracy{9})


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