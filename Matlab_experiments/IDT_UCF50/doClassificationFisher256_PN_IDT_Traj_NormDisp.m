% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVidTraj_IDT_N;
descParam.BlockSize = [8 8 6];
descParam.NumBlocks = [3 3 2];
descParam.MediaType = 'IDT';
descParam.NumOr = 8;
descParam.Normalisation='disp_traj';
%descParam.FrameSampleRate = 1;
%descParam.ColourSpace = colourSpace
descParam.IDTfeature='Traj';

%sRow = [1 3];
%sCol = [1 1];

% pcaDim & vocabulary size
pcaDim = 15;
numClusters = 256;

%%%%%%%%%%
bazePathFeatures='/data/MM31/iduta/Features/UCF50/IDT/Videos/'; %change

vocabularyIms = GetVideosPlusLabels('smallEnd');
vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '.gz'];
end
    
[gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
                                                numClusters, pcaDim);
%%%%%%%%

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');

pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i)) '.gz'];
end


% Now object visual word frequency histograms
fprintf('Fisher Kernel extraction for %d vids: ', length(pathFeatures));
for i=1:length(pathFeatures)
    if mod(i,100) == 0
        fprintf('%d ', i);
    end
    % Extract descriptors
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
    desc = desc'; 
    
    % Feature vector assignment with spatial pyramid
%     featSpIdx = SpatialPyramidSeparationIdx(info, sRow, sCol)';
%     fisherVT = cell(1,size(featSpIdx,1));
%     for spIdx = 1:size(featSpIdx,1)
%         %fisherVT{spIdx} = NormalizeRowsUnit(SquareRootAbs(mexFisherAssign(desc(:,featSpIdx(spIdx,:)), gmmModelName)'));
%         fisherVT{spIdx} = NormalizeRowsUnit(PowerNormalization(mexFisherAssign(desc(:,featSpIdx(spIdx,:)), gmmModelName)', 0.14));
%     end
%     fisherV = cat(2, fisherVT{:});
    
    % memory allocation
    if i == 1
        descParamUsed
        fisherV=NormalizeRowsUnit(PowerNormalization(mexFisherAssign(desc, gmmModelName)', 0.14));
        fisherVectors = zeros(length(pathFeatures), length(fisherV));
        fisherVectors(i,:)=fisherV;
    else
        fisherVectors(i,:) = NormalizeRowsUnit(PowerNormalization(mexFisherAssign(desc, gmmModelName)', 0.14));
    end
    
end
fprintf('\nDone!\n');

%% Do classification

% Histogram Intersection kernel
allDist = fisherVectors * fisherVectors';

% Leave-one-group-out cross-validation
for i=1:max(groups)
    testI = groups == i;
    trainI = ~testI;
    trainDist = allDist(trainI, trainI);
    testDist = allDist(testI, trainI);
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs);
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

perGroupAccuracy = mean(cat(2, accuracy{:}))'


saveName = [DATAopts.resultsPath DescParam2Name(descParam) 'Fisher.mat'];
save(saveName, '-v7.3', 'descParam', 'clfsOut', 'accuracy');
