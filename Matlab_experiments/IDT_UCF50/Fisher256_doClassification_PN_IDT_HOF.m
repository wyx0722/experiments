% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVidHOF_IDT;
descParam.BlockSize = [8 8 6];
descParam.NumBlocks = [3 3 2];
descParam.MediaType = 'IDT';
descParam.NumOr = 8;
%descParam.FrameSampleRate = 1;
%descParam.ColourSpace = colourSpace
descParam.IDTfeature='HOF';

%sRow = [1 3];
%sCol = [1 1];

% pcaDim & vocabulary size
pcaDim = 72;
numClusters = 256;

%%%%%%%%%%
bazePathFeatures='/home/ionut/Features/Features/UCF50/IDT/Videos/'; %change

vocabularyIms = GetVideosPlusLabels('smallEnd');
vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '.gz'];
end
    
[gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
                                                numClusters, pcaDim);

                                     


% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');

pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i)) '.gz'];
end


useSP=0;  %change to zero if Spatial Pyramid is not used

if (useSP==1)
    nDivImg=sRow(1)*sRow(2)*sCol(1)*sCol(2);
    if nDivImg>1
        nDivImg=nDivImg+1;
    end

    dimVlad = numClusters * pcaDim * nDivImg;

else
    dimVlad=numClusters * pcaDim;
end

nEncoding=1;

fisherVectors1= zeros(length(vids), 2*dimVlad);  


parpool(5);
% Now object visual word frequency histograms
fprintf('IDT FisherVector extraction  for %d vids: ', length(pathFeatures));
for i=1:length(pathFeatures)
    
    fprintf('%d', i)

    % Extract descriptors
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
    desc = desc';
    
    fisherVectors1(i,:)=mexFisherAssign(desc, gmmModelName)';

  
       
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');

%% Do classification

allDist=cell(1, nEncoding);

n_fisherVectors1=NormalizeRowsUnit(PowerNormalization(fisherVectors1, 0.14));
allDist{1}=n_fisherVectors1 * n_fisherVectors1';
clear n_fisherVectors1




all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

% cRange = 100;
% nReps = 1;
% nFolds = 3;


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
    
    [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs);
    %[~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

all_clfsOut{k}=clfsOut;
all_accuracy{k}=accuracy;

k
perGroupAccuracy = mean(cat(2, accuracy{:}))'

end

delete(gcp('nocreate'))

saveName = [DATAopts.resultsPath DescParam2Name(descParam) 'fisher256.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

