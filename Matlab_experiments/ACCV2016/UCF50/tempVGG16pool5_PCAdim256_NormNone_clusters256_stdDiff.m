
global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='TempVGG16Split1';
descParam.Normalisation='None'; %'ROOTSIFT';

descParam.numClusters = 256;

descParam.pcaDim = 256;

descParam

%%%%%%%%%%
bazePathFeatures='/home/ionut/Data/action_temporal_vgg_16_split1_features_opticalFlow_tvL1_UCF50/Videos/' %change


vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '/pool5.txt'];
end


                                           
[vocabulary, pcaMap, st_d, skew, nElem, kurt] = CreateVocabularyKmeansPca_m(vocabularyImsPaths, descParam, ...
                                                        descParam.numClusters, descParam.pcaDim);




%vocabulary = NormalizeRowsUnit(vocabulary); %make unit length

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i)) '/pool5.txt'];
end



    [tDesc] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
    %tDesc=NormalizeRowsUnit(tDesc);
    tVLAD=VLAD_1(tDesc, vocabulary);

vladNoMean=zeros(length(vids), length(tVLAD), 'like', tVLAD);  
stdDiff=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vladMean=zeros(length(vids), length(tVLAD), 'like', tVLAD);  
stdDiffMean=zeros(length(vids), length(tVLAD), 'like', tVLAD); 

parpool(13);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
parfor i=1:length(pathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
   % desc = NormalizeRowsUnit(desc);
   
    
    vladNoMean(i, :)=VLAD_1(desc, vocabulary);
    vladMean(i, :)=VLAD_1_mean(desc, vocabulary);
    stdDiff(i, :)=stdDiff_VLAD(desc, vocabulary, st_d);
    stdDiffMean(i, :)=stdDiff_VLAD_mean(desc, vocabulary, st_d);

         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=4;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(vladNoMean);
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(vladMean);
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(stdDiff);
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(stdDiffMean);
allDist{4}=temp * temp';

clear temp


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

mean(perGroupAccuracy)

end

delete(gcp('nocreate'))


% saveName = ['/home/ionut/Data/results/ICPR2016_rezults/' 'clfsOut/' 'encoding/'  DescParam2Name(descParam) '_vladNoMean_maxEncode_fisherVectors_.mat'];
% save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');
% 
%  saveName2 = ['/home/ionut/Data/results/ICPR2016_rezults/' 'videoRep/' 'encoding/' DescParam2Name(descParam) '_vladNoMean_maxEncode_fisherVectors_.mat'];
%  save(saveName2, '-v7.3', 'vladNoMean', 'maxEncode', 'fisherVectors');
%  
 

