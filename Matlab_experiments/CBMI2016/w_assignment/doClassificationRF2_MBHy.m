t=tic;
global DATAopts;
DATAopts = UCFInit;
% DATAopts.vocabularyPath = DATAopts.vocabularyPath;

% Settings for feature extraction
clear descParam

%descParam.Func = @FEVidHogDense;

descParam.Func = @FEVidMBHyDense;

%descParam.Func = @FEVidMBHxDense;
%descParam.Func = @FEVidMBHYDense;

descParam.BlockSize = [8 8 6];
descParam.NumBlocks = [3 3 2];
descParam.MediaType = 'Vid';
descParam.NumOr = 8;
descParam.FrameSampleRate = 1;
descParam.Normalisation='ROOTSIFT';

sRow = [1 3];
sCol = [1 1];

%%
% clear descParam
% descParam.Func = @FEVidLaptev;
% descParam.MediaType = 'Load';
% descParam.Type = 'Hof';

% Settings for visual vocabulary (including PCA reduction
pcaDim = 72;

%% Create Visual Vocabulary
[smallVids, smallLabs] = GetVideosPlusLabels('smallEnd');
[descName, saveNameRF, pcaMap] = CreateVocabularyRandomForestPca(smallVids, smallLabs, descParam, pcaDim, ...
    4, 10, pcaDim/2, 3000000, 0);


%% Now get visual word histograms
[vids, labs, groups] = GetVideosPlusLabels('Full');

% Load visual vocabulary
load(saveNameRF)
nTrees = length(maps); % Get number of trees

% Now object visual word frequency histograms
fprintf('VW creation for %d vids: ', length(vids));
for i=1:length(vids)
    if mod(i,100) == 0
        fprintf('%d ', i);
    end
    % Extract descriptors
    [desc, info, descParamUsed] = MediaName2Descriptor(vids{i}, descParam, pcaMap);
    
    % Project using random forest
    desc = desc'; % We need transposed descriptors
    cbEntry = cell(1, nTrees);
    spIdx = SpatialPyramidSeparationIdx(info, sRow, sCol);
    cbI = 1;
    for spI=1:size(spIdx,2)
        for j=1:nTrees
            cbEntry{cbI} = mexTreeAssign(desc(:,spIdx(:,spI)), maps{j}, boundaries{j});
            cbI = cbI + 1;
        end
    end

    if i == 1
        cbEntryT = cat(2, cbEntry{:});
        vwFrequencies = zeros(length(vids), size(cbEntryT, 2));
        clear cbEntryT;
        descParamUsed % Sanity check
    end
    
    % Store visual word frequency histogram per video
    vwFrequencies(i,:) = cat(2, cbEntry{:});    
end
fprintf('\n');

n_vwFrequencies=NormalizeRows(sqrt(vwFrequencies));
%% Do classification

% Histogram Intersection kernel
allDist = HistogramIntersectionSelf(n_vwFrequencies);
allDist = SetDiagonal(allDist, 1);

cRange = 100;
nReps = 1;
nFolds = 3;

% Leave-one-group-out cross-validation
for i=1:max(groups)
    testI = groups == i;
    trainI = ~testI;
    trainDist = allDist(trainI, trainI);
    testDist = allDist(testI, trainI);
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

perGroupAccuracy = mean(cat(2, accuracy{:}))'

mean(perGroupAccuracy)

saveName = [DATAopts.resultsPath DescParam2Name(descParam) 'RF.mat'];
save(saveName, '-v7.3', 'descParam', 'clfsOut', 'accuracy');

saveName2 = ['/home/ionut/Data/results/CBMI2015_rezults/' 'videoRep/' 'wAssig/' DescParam2Name(descParam) '_RF_.mat'];
save(saveName2, '-v7.3', 'vwFrequencies');
tf=toc(t);
