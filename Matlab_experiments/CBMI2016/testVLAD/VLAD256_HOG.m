

global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVidHofDense;
descParam.BlockSize = [8 8 6];
descParam.NumBlocks = [3 3 2];
descParam.MediaType = 'Vid';
descParam.NumOr = 8;
descParam.Normalisation='ROOTSIFT';


sRow = [1 3];
sCol = [1 1];

descParam.numClusters=256;
descParam.pcaDim=72;


descParam



vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=sprintf(DATAopts.videoPath, vocabularyIms{i});
end



                                            
[vocabulary, pcaMap] = CreateVocabularyKmeansPca(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim); 
                                            
%vocabulary = NormalizeRowsUnit(vocabulary); %make unit length

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
fullPathVids=cell(size(vids));

for i=1:length(fullPathVids)
    fullPathVids{i}=sprintf(DATAopts.videoPath, vids{i});
end



    [tDesc] = MediaName2Descriptor(fullPathVids{1}, descParam, pcaMap);
    tDesc=NormalizeRowsUnit(tDesc);
    tVLAD=VLAD_1_mean_fast(tDesc, vocabulary);
    
vlad1=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad2=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad3=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad4=zeros(length(vids), length(tVLAD), 'like', tVLAD);



% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(fullPathVids));
for i=1:length(fullPathVids)
    fprintf('%d ', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(fullPathVids{i}, descParam, pcaMap);
    %desc = NormalizeRowsUnit(desc);
    
        % Feature vector assignment with spatial pyramid
    featSpIdx = SpatialPyramidSeparationIdx(info, sRow, sCol)';
    
    
    vlad1(i, :)=VLAD_1(desc(featSpIdx(1,:), :), vocabulary);
    vlad2(i, :)=VLAD_1(desc(featSpIdx(2,:), :), vocabulary);
    vlad3(i, :)=VLAD_1(desc(featSpIdx(3,:), :), vocabulary);
    vlad4(i, :)=VLAD_1(desc(featSpIdx(4,:), :), vocabulary);
    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=1;
allDist=cell(1, nEncoding);

VLADAll=cat(2, vlad1,vlad2, vlad3, vlad4);


n_VLADAll=NormalizeRowsUnit(PowerNormalization(VLADAll, 0.5));
allDist{1}=n_VLADAll * n_VLADAll';
clear n_VLADAll

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


for k=1:nEncoding

% 
% Leave-one-group-out cross-validation
for i=1:max(groups)
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

mean(perGroupAccuracy)

end







saveName = [DATAopts.resultsPath DescParam2Name(descParam) 'VLAD.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = ['/home/ionut/Data/results/CBMI2015_rezults/' 'videoRep/' 'wAssig/' DescParam2Name(descParam) '_VLAD_.mat'];
 save(saveName2, '-v7.3', 'VLADAll');
