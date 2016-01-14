% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;
       

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');



avg_conv4_3=zeros(length(vids), 28*28*512);


max_conv4_3=zeros(length(vids), 28*28*512);

rootPathFeatures='/home/ionut/Data/VGG_16_v-features_rawFrames_UCF50/Videos/';

parpool(5);

fprintf('Load VGG features for %d vids: ', length(vids));
parfor i=1:length(vids)
    
        fprintf('%d ', i);
 
    
    % Extract descriptors   

   [avg_conv4_3(i, :),max_conv4_3(i, :) ]=avg_max_FeaturesVideo([rootPathFeatures, vids{i}, '/conv4_3.txt'], 'conv4_3');
 
      
end
fprintf('\nDone!\n');

%% Do classification

% Histogram Intersection kernel
%allDist = vladVectors * vladVectors';



nEncoding=2;

allDist=cell(1, nEncoding);

n_avg_conv4_3=NormalizeRowsUnit(SquareRootAbs(avg_conv4_3));
allDist{1}=n_avg_conv4_3 * n_avg_conv4_3';
clear n_avg_conv4_3

n_max_conv4_3=NormalizeRowsUnit(SquareRootAbs(max_conv4_3));
allDist{2}=n_max_conv4_3 * n_max_conv4_3';
clear n_max_conv4_3



all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);



cRange = 100;
nReps = 1;
nFolds = 3;



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
perGroupAccuracy = mean(cat(2, accuracy{:}))'

end

delete(gcp('nocreate'))



% saveName = [DATAopts.resultsPath DescParam2Name(descParam) 'VLAD_1_mean_512C.mat'];
% save(saveName, '-v7.3', 'descParam', 'clfsOut', 'accuracy', 'vectNrFrames', 'tFinalWorldA');

% saveName2 = [DATAopts.featurePath DescParam2Name(descParam) 'VLADE_1_512C.mat'];
% save(saveName2, '-v7.3', 'descParam', 'vladVectors', 'groups', 'labs');