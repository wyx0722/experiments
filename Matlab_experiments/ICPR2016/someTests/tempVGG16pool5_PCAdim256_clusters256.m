
global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='TempVGG16';
descParam.Normalisation='ROOTSIFT';

descParam.numClusters = 256;

descParam.pcaDim = 256;

descParam

%%%%%%%%%%
bazePathFeatures='/home/ionut/Data/action_temporal_vgg_16_split1_features_opticalFlow_tvL1_UCF50/Videos/'; %change


vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '/pool5.txt'];
end


vocabs=cell(1, 2);
pcaMaps=cell(1, 2);


parpool(2);
parfor i=1:2
    if i==1                                            
        [vocabs{i}, pcaMaps{i}] = CreateVocabularyKmeansPca(vocabularyImsPaths, descParam, ...
                                                        descParam.numClusters, descParam.pcaDim);
    else
        [vocabs{i}, pcaMaps{i}] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
                                                        descParam.numClusters, descParam.pcaDim);
    end

end
delete(gcp('nocreate'))

pcaMap=pcaMaps{1};
vocabulary=vocabs{1};
gmmModelName=vocabs{2};




%vocabulary = NormalizeRowsUnit(vocabulary); %make unit length

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i)) '/pool5.txt'];
end



    [tDesc] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
    %tDesc=NormalizeRowsUnit(tDesc);
    tVLAD=VLAD_1_mean(tDesc, vocabulary);

vladNoMean=zeros(length(vids), length(tVLAD), 'like', tVLAD);    
vlad=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad1=zeros(length(vids), length(tVLAD), 'like', tVLAD);
avgEncode=zeros(length(vids), length(tVLAD), 'like', tVLAD);
maxEncode=zeros(length(vids), length(tVLAD), 'like', tVLAD);

[tDesc] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
FV=mexFisherAssign(tDesc', gmmModelName)';

fisherVectors=zeros(length(vids), length(FV), 'like', FV);

parpool(24);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
parfor i=1:length(pathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
   % desc = NormalizeRowsUnit(desc);
   
    
    vladNoMean(i, :)=VLAD_1(desc, vocabulary);
    vlad(i, :)=VLAD_1_mean(desc, vocabulary);
    vlad1(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 1);
    [avgEncode(i, :), maxEncode(i, :)]=avg_max_pooling(desc, vocabulary);
    
    fisherVectors(i,:)=mexFisherAssign(desc', gmmModelName)';
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=6;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(PowerNormalization(vladNoMean, 0.5));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(vlad, 0.5));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(vlad1, 0.5));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(avgEncode, 0.5));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(maxEncode, 0.5));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(fisherVectors, 0.5));
allDist{6}=temp * temp';

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







