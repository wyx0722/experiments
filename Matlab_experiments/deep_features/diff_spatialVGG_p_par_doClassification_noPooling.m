% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_conv5_3_diff_noPool;
descParam.BlockSize = [8 8 6];
descParam.NumBlocks = [3 3 2];
descParam.MediaType = 'DeepF';
descParam.NumOr = 8;
descParam.FrameSampleRate = 1;
%descParam.ColourSpace = colourSpace
descParam.Layer='conv5_3';
descParam.Net='spatialVGG16';
descParam.nPool=10;

sRow = [1 3];
sCol = [1 1];

% pcaDim & vocabulary size
pcaDim = 256;

numClusters = 256;

vocabularyIms = GetVideosPlusLabels('smallEnd');
vocabularyImsPaths=cell(size(vocabularyIms));

rootPathFeatures='/home/ionut/Data/VGG_16_v-features_rawFrames_UCF50/Videos/';
for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[rootPathFeatures char(vocabularyIms(i)) '/conv5_3.txt'];
end

 %%%%%%%%%%%%%%%%%%%%%%%%                                           
 [vocabulary, pcaMap, st_d, skew, nElem] = CreateVocabularyKmeansPca_m(vocabularyImsPaths, descParam, numClusters, pcaDim);
%vocabulary = NormalizeRowsUnit(vocabulary);
 %%%%%%%%%%%%%%%%%%%%%%%%                                          
                                          
                                                    
% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');

pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[rootPathFeatures char(vids(i)) '/conv5_3.txt'];
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




% Now object visual word frequency histograms
fprintf('Feature extraction for %d vids: ', length(pathFeatures));


parpool(13);
parfor i=1:length(pathFeatures)
%     if mod(i,100) == 0
%         fprintf('%d ', i);
%     end
    fprintf('\b%d\n', i)
    % Extract descriptors   
    %%%%%%%%%%%%%%%%%%%%%%%
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
    %desc = NormalizeRowsUnit(desc); %!!!!! it left here, possible mistake
    %%%%%%%%%%%%%%%%%%%
     
   
 
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

n_vladVectors1=NormalizeRowsUnit(SquareRootAbs(vladVectors1));
allDist{1}=n_vladVectors1 * n_vladVectors1';
clear n_vladVectors1

n_vladVectors2=NormalizeRowsUnit(SquareRootAbs(vladVectors2));
allDist{2}=n_vladVectors2 * n_vladVectors2';
clear n_vladVectors2

n_vladVectors3=NormalizeRowsUnit(SquareRootAbs(vladVectors3));
allDist{3}=n_vladVectors3 * n_vladVectors3';
clear n_vladVectors3



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