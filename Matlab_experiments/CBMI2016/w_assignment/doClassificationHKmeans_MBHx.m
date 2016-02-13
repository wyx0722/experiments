
global DATAopts;
DATAopts = UCFInit;
DATAopts.vocabularyPath = DATAopts.vocabularyPath;

% Settings for feature extraction
clear descParam
descParam.Func = @FEVidMBHxDense;
descParam.BlockSize = [8 8 6];
descParam.NumBlocks = [3 3 2];
descParam.MediaType = 'Vid';
descParam.NumOr = 8;
descParam.Normalisation='ROOTSIFT';

%%
% clear descParam
% descParam.Func = @FEVidLaptev;
% descParam.MediaType = 'Load';
% descParam.Type = 'Hof';

sRow = [1 3];
sCol = [1 1];

% Settings for visual vocabulary (including PCA reduction
pcaDim = 72;
numClusters = 64;
depth = 2;

%% Create Visual Vocabulary
[smallVids, smallLabs] = GetVideosPlusLabels('smallEnd');
[kmeansTree, pcaMap, minDesc, rangeDesc] = ...
    CreateVocabularyKmeansHierarachicalPca(smallVids, descParam, ...
                                                numClusters, depth, pcaDim);

%% Now get visual word histograms
[vids, labs, groups] = GetVideosPlusLabels('Full');

% Now object visual word frequency histograms
fprintf('VW creation for %d vids: ', length(vids));

tFinalWorldA=zeros(1, length(vids));
vectNrFrames=zeros(1, length(vids));


for i=1:length(vids)
    if mod(i,100) == 0
        fprintf('%d ', i);
    end
    % Extract descriptors
    [desc, info, descParamUsed, nrFrames] = MediaName2Descriptor(vids{i}, descParam, pcaMap);
    vectNrFrames(i)=nrFrames;
    
	
	tStartWorldA=tic;
	% Project using random forest
    featSpIdx = SpatialPyramidSeparationIdx(info, sRow, sCol);
    cbEntry = cell(1,size(featSpIdx,2));
    for spIdx = 1:size(featSpIdx,2)
        cbEntry{spIdx} = KmeansHierarchicalAssignment(desc(featSpIdx(:,spIdx),:), kmeansTree, minDesc, rangeDesc);
    end
    
    if i == 1
        cbEntryT = cat(2, cbEntry{:});
        vwFrequencies = zeros(length(vids), size(cbEntryT, 2));
        clear cbEntryT;
        descParamUsed % Sanity check
    end
    
    % Store visual word frequency histogram per video
    %vwFrequencies(i,:) = NormalizeRows(sqrt(cat(2, cbEntry{:})));
    vwFrequencies(i,:) = cat(2, cbEntry{:});
	
	tFinalWorldA(i)=toc(tStartWorldA);
	
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
    
    [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs,cRange, nReps, nFolds);
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

perGroupAccuracy = mean(cat(2, accuracy{:}))'

mean(perGroupAccuracy)

saveName = [DATAopts.resultsPath DescParam2Name(descParam) 'HKmeans.mat'];
save(saveName, '-v7.3', 'descParam', 'clfsOut', 'accuracy', 'vectNrFrames', 'tFinalWorldA');

 saveName2 = ['/home/ionut/Data/results/CBMI2015_rezults/' 'videoRep/' 'wAssig/' DescParam2Name(descParam) '_HKmeans_.mat'];
 save(saveName2, '-v7.3', 'vwFrequencies');
