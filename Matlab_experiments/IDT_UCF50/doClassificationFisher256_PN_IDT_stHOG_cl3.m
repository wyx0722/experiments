% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_stHOG_IDT;
%descParam.BlockSize = [8 8 6];
%descParam.NumBlocks = [3 3 2];
descParam.MediaType = 'IDT';
descParam.NumOr = 8;
%descParam.FrameSampleRate = 1;
%descParam.ColourSpace = colourSpace
descParam.IDTfeature='HOG';

%sRow = [1 3];
%sCol = [1 1];

% pcaDim & vocabulary size
pcaDim = 72;
numClusters = 256;

%%%%%%%%%%
bazePathFeatures='/data/MM31/iduta/Features/UCF50/STHOG_IDT/Videos/'; %change

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

emptyCluster={};

% Now object visual word frequency histograms
fprintf('Fisher Kernel extraction for %d vids: ', length(pathFeatures));
for i=1:length(pathFeatures)
    if mod(i,100) == 0
        fprintf('%d ', i);
    end
    % Extract descriptors
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
    desc = desc'; 
    
    
    nrCl=3;
   
    [idx, statusEmptyCluster] = Cluster3(info.infoTraj, nrCl);

    if statusEmptyCluster==1      
        emptyCluster=[emptyCluster i];      
    end
    
    % Feature vector assignment with spatial pyramid
%     featSpIdx = SpatialPyramidSeparationIdx(info, sRow, sCol)';
%     fisherVT = cell(1,size(featSpIdx,1));
%     for spIdx = 1:size(featSpIdx,1)
%         %fisherVT{spIdx} = NormalizeRowsUnit(SquareRootAbs(mexFisherAssign(desc(:,featSpIdx(spIdx,:)), gmmModelName)'));
%         fisherVT{spIdx} = NormalizeRowsUnit(PowerNormalization(mexFisherAssign(desc(:,featSpIdx(spIdx,:)), gmmModelName)', 0.14));
%     end
%     fisherV = cat(2, fisherVT{:});
    
    % memory allocation
 
    
     fisherVT = cell(1,nrCl+1);
    
    fisherVT{1} = NormalizeRowsUnit(PowerNormalization(mexFisherAssign(desc(:,:), gmmModelName)', 0.14));
    
    for c=1:nrCl
        
        fisherVT{c+1}=NormalizeRowsUnit(PowerNormalization(mexFisherAssign(desc(:,idx==c), gmmModelName)', 0.14));
     
    end
    
    
    
    fisherV = cat(2, fisherVT{:}); 
    
    % memory allocation
    if i == 1
        descParamUsed
        fisherVectors = zeros(length(vids), length(fisherV));
    end
    
    fisherVectors(i,:) = fisherV;
    
end
fprintf('\nDone!\n');

%% Do classification

% Histogram Intersection kernel
allDist = fisherVectors * fisherVectors';

% Leave-one-group-out cross-validation
matlabpool(5);

parfor i=1:max(groups)
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


saveName = [DATAopts.resultsPath DescParam2Name(descParam) 'Fisher_cl3.mat'];
save(saveName, '-v7.3', 'descParam', 'clfsOut', 'accuracy');

matlabpool close