global DATAopts;
DATAopts = HMDB51Init;

spl{1}='HMBD51Split1';
spl{2}='HMBD51Split2';
spl{3}='HMBD51Split3';

for s=1:3
    
    
% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='SpVGG19';
descParam.Normalisation='None';
descParam.numClusters = 256;
descParam.pcaDim = 0;
descParam.Dataset=spl(s);
descParam

[allVids, labs, splits] = GetVideosPlusLabels();

%the baze path for features
bazePathFeatures='/home/ionut/asustor_ionut/ionut/Data/hmdb51_VGG_19_features_rawFrames/Videos/' %change

%create the full path of the fetures for each video
allPathFeatures=cell(size(allVids));
for i=1:size(allVids, 1)
    allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4) '/pool5.txt'];
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

[gmmModelName, pcaMap2] = CreateVocabularyGMMPca(vocabularyPathFeatures, descParam, ...
                                                        descParam.numClusters, descParam.pcaDim);
                                            
                                            
[tDesc] = MediaName2Descriptor(trainTestSetPathFeatures{1}, descParam, pcaMap);                                           
tVLAD=VLAD_1(tDesc, vocabulary, 1);

vladNoMean=zeros(length(trainTestSetPathFeatures), length(tVLAD), 'like', tVLAD);
maxEncode=zeros(length(trainTestSetPathFeatures), length(tVLAD), 'like', tVLAD);

[tDesc] = MediaName2Descriptor(trainTestSetPathFeatures{1}, descParam, pcaMap);
FV=mexFisherAssign(tDesc', gmmModelName)';

fisherVectors=zeros(length(trainTestSetPathFeatures), length(FV), 'like', FV);


fprintf('Feature extraction  for %d vids: ', length(trainTestSetPathFeatures));
parpool(10);
for i=1:length(trainTestSetPathFeatures)
    fprintf('%d \n', i)
    
    [desc, info, descParamUsed] = MediaName2Descriptor(trainTestSetPathFeatures{i}, descParam, pcaMap);
    
    
     vladNoMean(i, :)=VLAD_1(desc, vocabulary);
     [~, maxEncode(i, :)]=avg_max_pooling(desc, vocabulary);
     
     fisherVectors(i,:)=mexFisherAssign(desc', gmmModelName)';
     
   
        
     if i == 1
         descParamUsed
     end
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

%% Do classification

nEncoding=3;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(vladNoMean);
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(maxEncode);
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(fisherVectors);
allDist{1}=temp * temp';

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
    trainLabs = trainTestSetlabs(trainI,:);
    testLabs = trainTestSetlabs(testI, :);
    
    [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy = ClassificationAccuracy(clfsOut, testLabs);
    fprintf('accuracy: %.3f\n', accuracy);
    
    all_clfsOut{k}=clfsOut;
    all_accuracy{k}=accuracy;
end

delete(gcp('nocreate'))

acc1=mean(all_accuracy{1})
acc2=mean(all_accuracy{2})
acc3=mean(all_accuracy{3})



fileName=sprintf('/home/ionut/experiments/Matlab_experiments/ACMMM2016/results/results_HMDB51_Features%s_Layer%s_Network%s_PCAdim%d_clusters%d_norm%s__VLAD.txt', descParam.MediaType,descParam.Layer, descParam.net,descParam.pcaDim, descParam.numClusters, descParam.Normalisation); 
fileID=fopen(fileName, 'a');

fprintf(fileID, 'Dataset and Split: %s --> vladNoMean acc= %.4f   maxEncode acc= %.4f  fisherVectors acc= %.4f \r\n', descParam.Dataset, acc1,acc2, acc3 );
fclose(fileID);


saveName = ['/home/ionut/Data/results/ICPR2016_rezults/' 'clfsOut/' 'encoding/'  DescParam2Name(descParam) '_vladNoMean_maxEncode_fisherVectors_.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = ['/home/ionut/Data/results/ICPR2016_rezults/' 'videoRep/' 'encoding/' DescParam2Name(descParam) '_vladNoMean_maxEncode_fisherVectors_.mat'];
 save(saveName2, '-v7.3', 'vladNoMean', 'maxEncode', 'fisherVectors');
 

 

end