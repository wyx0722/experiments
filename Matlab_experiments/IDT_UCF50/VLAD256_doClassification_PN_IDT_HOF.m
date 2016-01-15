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
    
%[gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
%                                                numClusters, pcaDim);
[vocabulary, pcaMap, st_d, skew, nElem] = CreateVocabularyKmeansPca_m(vocabularyImsPaths, descParam, ...
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

    dimVlad = size(vocabulary, 1) * size(vocabulary, 2) * nDivImg;

else
    dimVlad=size(vocabulary, 1) * size(vocabulary, 2);
end

nEncoding=3;

vladVectors1= zeros(length(vids), dimVlad);  
vladVectors2= zeros(length(vids), 2*dimVlad);
vladVectors3= zeros(length(vids), 3*dimVlad);


parpool(5);
% Now object visual word frequency histograms
fprintf('IDT VLAD extraction  for %d vids: ', length(pathFeatures));
parfor i=1:length(pathFeatures)
    fprintf('\b%d\n', i)
    % Extract descriptors
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
    
        vladVectors1(i,:)=VLAD_1_mean_L2(desc, vocabulary);
        vladVectors2(i,:)=improvedVLAD_intraL2(desc, vocabulary, st_d, skew, nElem);
        vladVectors3(i,:)=BoostingVLAD_paper_intraL2(desc, vocabulary, st_d, skew, nElem);
   
  
       
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');

%% Do classification

allDist=cell(1, nEncoding);

n_vladVectors1=NormalizeRowsUnit(PowerNormalization(vladVectors1, 0.14));
allDist{1}=n_vladVectors1 * n_vladVectors1';
clear n_vladVectors1

n_vladVectors2=NormalizeRowsUnit(PowerNormalization(vladVectors2, 0.14));
allDist{2}=n_vladVectors2 * n_vladVectors2';
clear n_vladVectors2

n_vladVectors3=NormalizeRowsUnit(PowerNormalization(vladVectors3, 0.14));
allDist{3}=n_vladVectors3 * n_vladVectors3';
clear n_vladVectors3



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


saveName = [DATAopts.resultsPath DescParam2Name(descParam) 'VLAD256.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');
