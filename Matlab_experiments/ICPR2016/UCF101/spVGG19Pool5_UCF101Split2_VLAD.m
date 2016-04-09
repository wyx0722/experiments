global DATAopts;
DATAopts = UCF101Init;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='SpVGG19';
descParam.Normalisation='ROOTSIFT';
descParam.numClusters = 256;
descParam.pcaDim = 256;
descParam.Dataset='UCF101';
descParam.Split=2;
descParam

[allVids, labs, splits] = GetVideosPlusLabels('Challenge');

%the baze path for features
bazePathFeatures='/home/ionut/halley_ionut/Data/VGG_19_features_rawFrames_UCF101/Videos/'; %change

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


                                            
                                            
[tDesc] = MediaName2Descriptor(allPathFeatures{1}, descParam, pcaMap);                                           
tVLAD=doubleAssign_VLAD_1(tDesc, vocabulary, 1);

vlad1=zeros(length(allPathFeatures), length(tVLAD), 'like', tVLAD);
vlad2=zeros(length(allPathFeatures), length(tVLAD), 'like', tVLAD);
vlad3=zeros(length(allPathFeatures), length(tVLAD), 'like', tVLAD);

fprintf('Feature extraction  for %d vids: ', length(allPathFeatures));
parpool(12);
parfor i=1:length(allPathFeatures)
    fprintf('%d \n', i)
    
    [desc, info, descParamUsed] = MediaName2Descriptor(allPathFeatures{i}, descParam, pcaMap);
    
    vlad1(i,:)=VLAD_1_mean(desc, vocabulary);
    vlad2(i,:)=doubleAssign_VLAD_1(desc, vocabulary, 0.5); %!!!!!!!!
    vlad3(i,:)=doubleAssign_VLAD_1(desc, vocabulary, 1); %!!!!!!!!
   
        
     if i == 1
         descParamUsed
     end
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

%% Do classification

nEncoding=3;
allDist=cell(1, nEncoding);

n_vlad1=NormalizeRowsUnit(PowerNormalization(vlad1, 0.5));
allDist{1}=n_vlad1 * n_vlad1';
clear n_vlad1

n_vlad2=NormalizeRowsUnit(PowerNormalization(vlad2, 0.5));
allDist{2}=n_vlad2 * n_vlad2';
clear n_vlad2

n_vlad3=NormalizeRowsUnit(PowerNormalization(vlad3, 0.5));
allDist{3}=n_vlad3 * n_vlad3';
clear n_vlad3


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
end

delete(gcp('nocreate'))

mean(all_accuracy{1})
mean(all_accuracy{2})
