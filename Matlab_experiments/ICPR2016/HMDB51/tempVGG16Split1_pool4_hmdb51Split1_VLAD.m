global DATAopts;
DATAopts = HMDB51Init;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool4';
descParam.net='tempSplit1VGG16';
descParam.Normalisation='ROOTSIFT';
descParam.numClusters = 256;
descParam.pcaDim = 256;
descParam.Dataset='HMBD51Split1';
descParam.DoubleAssign=int8(1); %!!!!!!!!!!
descParam

[allVids, labs, splits] = GetVideosPlusLabels();

%the baze path for features
bazePathFeatures='/home/ionut/halley_ionut/Data/hmdb51_action_temporal_vgg_16_split1_features_opticalFlow_tvL1/Videos/'; %change

%create the full path of the fetures for each video
allPathFeatures=cell(size(allVids));
for i=1:size(allVids, 1)
    allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4) '/pool4.txt'];
end


%get the data for a specific split
switch descParam.Dataset
    
    case 'HMBD51Split1'
        
        trainTestSetPathFeatures=allPathFeatures(splits(:, 1)==1 | splits(:, 1)==2);%get all the paths for the current split
        trainTestSetlabs=labs((splits(:, 1)==1 | splits(:, 1)==2), :); %get all the labels for the current split
        trainTestSplit=splits((splits(:, 1)==1 | splits(:, 1)==2), 1); %get the devision of date between training and testing set for the current split. Exclude the videos not included in the split (0 value)
        
        trainingSetPathFeatures=trainTestSetPathFeatures(trainTestSplit==1); %get the trining set feature paths 
        vocabularyPathFeatures=trainingSetPathFeatures(1:2:end); % build the vocabulary for half of the videos of the training set
        
    case 'HMBD51Split2'
     
        trainTestSetPathFeatures=allPathFeatures(splits(:, 2)==1 | splits(:, 2)==2);
        trainTestSetlabs=labs((splits(:, 2)==1 | splits(:, 2)==2), :);
        trainTestSplit=splits((splits(:, 2)==1 | splits(:, 2)==2), 2);
        
        trainingSetPathFeatures=trainTestSetPathFeatures(trainTestSplit==1);
        vocabularyPathFeatures=trainingSetPathFeatures(1:2:end);
        
    case 'HMBD51Split3'
        
        trainTestSetPathFeatures=allPathFeatures(splits(:, 3)==1 | splits(:, 3)==2);
        trainTestSetlabs=labs((splits(:, 3)==1 | splits(:, 3)==2), :);
        trainTestSplit=splits((splits(:, 3)==1 | splits(:, 3)==2), 3);
        
        trainingSetPathFeatures=trainTestSetPathFeatures(trainTestSplit==1);
        vocabularyPathFeatures=trainingSetPathFeatures(1:2:end);

    otherwise
        warning('Not known argument: Should be:  HMBD51Split1, HMBD51Split2 or HMBD51Split3 ')    
end


[vocabulary, pcaMap] = CreateVocabularyKmeansPca(vocabularyPathFeatures, descParam, ...
                                                descParam.numClusters, descParam.pcaDim); 


                                            
                                            
[tDesc] = MediaName2Descriptor(trainTestSetPathFeatures{1}, descParam, pcaMap);                                           
tVLAD=doubleAssign_VLAD_1(tDesc, vocabulary, 1);

vlad1=zeros(length(trainTestSetPathFeatures), length(tVLAD), 'like', tVLAD);
vlad2=zeros(length(trainTestSetPathFeatures), length(tVLAD), 'like', tVLAD);

fprintf('Feature extraction  for %d vids: ', length(trainTestSetPathFeatures));
parpool(24);
parfor i=1:length(trainTestSetPathFeatures)
    fprintf('%d \n', i)
    
    [desc, info, descParamUsed] = MediaName2Descriptor(trainTestSetPathFeatures{i}, descParam, pcaMap);
    
    vlad1(i,:)=VLAD_1_mean(desc, vocabulary);
    vlad2(i,:)=doubleAssign_VLAD_1(desc, vocabulary, 1); %!!!!!!!!
   
        
     if i == 1
         descParamUsed
     end
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

%% Do classification

nEncoding=2;
allDist=cell(1, nEncoding);

n_vlad1=NormalizeRowsUnit(PowerNormalization(vlad1, 0.5));
allDist{1}=n_vlad1 * n_vlad1';
clear n_vlad1

n_vlad2=NormalizeRowsUnit(PowerNormalization(vlad2, 0.5));
allDist{2}=n_vlad2 * n_vlad2';
clear n_vlad2


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
    trainLabs = trainTestSetlabs(trainI,:);
    testLabs = trainTestSetlabs(testI, :);
    
    [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy = ClassificationAccuracy(clfsOut, testLabs);
    fprintf('accuracy: %.3f\n', accuracy);
    
    all_clfsOut{k}=clfsOut;
    all_accuracy{k}=accuracy;
end

delete(gcp('nocreate'))

all_accuracy{1}
all_accuracy{2}
