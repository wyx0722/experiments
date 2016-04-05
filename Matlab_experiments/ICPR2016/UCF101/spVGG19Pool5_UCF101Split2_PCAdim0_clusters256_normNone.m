global DATAopts;
DATAopts = UCF101Init;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='SpVGG19';
descParam.Normalisation='None'; %'ROOTSIFT';
descParam.numClusters = 256;
descParam.pcaDim = 0;
descParam.Dataset='UCF101';
descParam.Split=2;
descParam

[allVids, labs, splits] = GetVideosPlusLabels('Challenge');

%the baze path for features
bazePathFeatures='/home/ionut/halley_ionut/Data/VGG_19_features_rawFrames_UCF101/Videos/' %change

%create the full path of the fetures for each video
allPathFeatures=cell(size(allVids));
for i=1:size(allVids, 1)
    allPathFeatures{i}=[bazePathFeatures allVids{i} '/pool5.txt'];
end

%get the data for a specific split
trainTestSplit=splits(:, descParam.Split); %get the devision of date between training and testing set for the current split. Exclude the videos not included in the split (0 value)
trainingSetPathFeatures=allPathFeatures(trainTestSplit==1); %get the trining set feature paths
vocabularyPathFeatures=trainingSetPathFeatures(1:3:end); % build the vocabulary from a third of the videos of the training set



[vocabulary, pcaMap] = CreateVocabularyKmeansPca(vocabularyPathFeatures, descParam, ...
                                                        descParam.numClusters, descParam.pcaDim);

[gmmModelName, pcaMap2] = CreateVocabularyGMMPca(vocabularyPathFeatures, descParam, ...
                                                        descParam.numClusters, descParam.pcaDim);


                                            
                                            
[tDesc] = MediaName2Descriptor(allPathFeatures{1}, descParam, pcaMap);                                           
tVLAD=VLAD_1_mean(tDesc, vocabulary);

vladNoMean=zeros(length(allPathFeatures), length(tVLAD), 'like', tVLAD);    
vlad=zeros(length(allPathFeatures), length(tVLAD), 'like', tVLAD);
vlad1=zeros(length(allPathFeatures), length(tVLAD), 'like', tVLAD);
avgEncode=zeros(length(allPathFeatures), length(tVLAD), 'like', tVLAD);
maxEncode=zeros(length(allPathFeatures), length(tVLAD), 'like', tVLAD);

[tDesc] = MediaName2Descriptor(allPathFeatures{1}, descParam, pcaMap);
FV=mexFisherAssign(tDesc', gmmModelName)';

fisherVectors=zeros(length(allPathFeatures), length(FV), 'like', FV);


fprintf('Feature extraction  for %d vids: ', length(allPathFeatures));
parpool(24);
parfor i=1:length(allPathFeatures)
    fprintf('%d \n', i)
    
    [desc, info, descParamUsed] = MediaName2Descriptor(allPathFeatures{i}, descParam, pcaMap);
    
    vladNoMean(i, :)=VLAD_1(desc, vocabulary);
    vlad(i, :)=VLAD_1_mean(desc, vocabulary);
    vlad1(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 1);
    [avgEncode(i, :), maxEncode(i, :)]=avg_max_pooling(desc, vocabulary);
    
    fisherVectors(i,:)=mexFisherAssign(desc', gmmModelName)';
   
        
     if i == 1
         descParamUsed
     end
end
delete(gcp('nocreate'))
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

parpool(nEncoding);
parfor k=1:nEncoding
    k
    trainI=trainTestSplit==1;
    testI=~trainI;
    
    trainDist = allDist{k}(trainI, trainI);
    testDist = allDist{k}(testI, trainI);
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy = ClassificationAccuracy(clfsOut, testLabs);
    fprintf('accuracy: %.3f\n', mean(accuracy));
    
    all_clfsOut{k}=clfsOut;
    all_accuracy{k}=accuracy;
    k
    mean(accuracy)
end

delete(gcp('nocreate'))

