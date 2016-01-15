% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');



avg_fc8=zeros(length(vids), 1000);
avg_fc7=zeros(length(vids), 4096);
avg_fc6=zeros(length(vids), 4096);
avg_conv5_3=zeros(length(vids), 14*14*512);
avg_conv5_1=zeros(length(vids), 14*14*512);

max_fc8=zeros(length(vids), 1000);
max_fc7=zeros(length(vids), 4096);
max_fc6=zeros(length(vids), 4096);
max_conv5_3=zeros(length(vids), 14*14*512);
max_conv5_1=zeros(length(vids), 14*14*512);

rootPathFeatures='/home/ionut/Data/VGG_16_v-features_rawFrames_UCF50/Videos/';

parpool(13);

fprintf('Load VGG features for %d vids: ', length(vids));
parfor i=1:length(vids)
   % if mod(i,100) == 0
    %    fprintf('%d ', i);
    %end
    
    fprintf('\b%d\n', i)

    % Extract descriptors   
   [avg_fc8(i, :), max_fc8(i, :)]=diff_avg_max_FeaturesVideo([rootPathFeatures, vids{i}, '/fc8.txt'], 'fc8_sVGG');
   [avg_fc7(i, :),max_fc7(i, :) ]=diff_avg_max_FeaturesVideo([rootPathFeatures, vids{i}, '/fc7.txt'], 'fc7');
   [avg_fc6(i, :), max_fc6(i, :)]=diff_avg_max_FeaturesVideo([rootPathFeatures, vids{i}, '/fc6.txt'], 'fc6');
   [avg_conv5_3(i, :),max_conv5_3(i, :) ]=diff_avg_max_FeaturesVideo([rootPathFeatures, vids{i}, '/conv5_3.txt'], 'conv5_3');
   [avg_conv5_1(i, :), max_conv5_1(i, :)]=diff_avg_max_FeaturesVideo([rootPathFeatures, vids{i}, '/conv5_1.txt'], 'conv5_1');



end
fprintf('\nDone!\n');

%% Do classification

% Histogram Intersection kernel
%allDist = vladVectors * vladVectors';


nEncoding=10;

allDist=cell(1, nEncoding);

n_avg_fc8=NormalizeRowsUnit(SquareRootAbs(avg_fc8));
allDist{1}=n_avg_fc8 * n_avg_fc8';
clear n_avg_fc8

n_avg_fc7=NormalizeRowsUnit(SquareRootAbs(avg_fc7));
allDist{2}=n_avg_fc7 * n_avg_fc7';
clear n_avg_fc7

n_avg_fc6=NormalizeRowsUnit(SquareRootAbs(avg_fc6));
allDist{3}=n_avg_fc6 * n_avg_fc6';
clear n_avg_fc6

n_avg_conv5_3=NormalizeRowsUnit(SquareRootAbs(avg_conv5_3));
allDist{4}=n_avg_conv5_3 * n_avg_conv5_3';
clear n_avg_conv5_3

n_avg_conv5_1=NormalizeRowsUnit(SquareRootAbs(avg_conv5_1));
allDist{5}=n_avg_conv5_1 * n_avg_conv5_1';
clear n_avg_conv5_1



n_max_fc8=NormalizeRowsUnit(SquareRootAbs(max_fc8));
allDist{6}=n_max_fc8 * n_max_fc8';
clear n_max_fc8

n_max_fc7=NormalizeRowsUnit(SquareRootAbs(max_fc7));
allDist{7}=n_max_fc7 * n_max_fc7';
clear n_max_fc7

n_max_fc6=NormalizeRowsUnit(SquareRootAbs(max_fc6));
allDist{8}=n_max_fc6 * n_max_fc6';
clear n_max_fc6

n_max_conv5_3=NormalizeRowsUnit(SquareRootAbs(max_conv5_3));
allDist{9}=n_max_conv5_3 * n_max_conv5_3';
clear n_max_conv5_3

n_max_conv5_1=NormalizeRowsUnit(SquareRootAbs(max_conv5_1));
allDist{10}=n_max_conv5_1 * n_max_conv5_1';
clear n_max_conv5_1




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






                                                                                                                                                                                          