global DATAopts;
DATAopts = HMDB51Init;




%the baze path for features
bazePathFeatures='/home/ionut/asustor_ionut_2/Data/ActionBank/ab_hmdb51_e1f1g2_matlab/'




[allVids, labs, splits] = GetVideosPlusLabels();



%create the full path of the fetures for each video
allPathFeatures=cell(size(allVids));
for i=1:size(allVids, 1)  
        allPathFeatures{i}=[bazePathFeatures allVids{i} '_banked.mat'];
end



trainTestSetPathFeatures=allPathFeatures(splits(:, descParam.Split)==1 | splits(:, descParam.Split)==2);%get all the paths for the current split
trainTestSetlabs=labs((splits(:, descParam.Split)==1 | splits(:, descParam.Split)==2), :); %get all the labels for the current split
trainTestSplit=splits((splits(:, descParam.Split)==1 | splits(:, descParam.Split)==2), 1); %get the devision of date between training and testing set for the current split. Exclude the videos not included in the split (0 value)

trainingSetPathFeatures=trainTestSetPathFeatures(trainTestSplit==1); %get the trining set feature paths 
%vocabularyPathFeatures=trainingSetPathFeatures(1:2:end); % build the vocabulary for half of the videos of the training set

trainLabs=trainTestSetlabs(trainTestSplit==1, :); %get the labels for the training set


                                                     

load(trainTestSetPathFeatures{1});
tRep=v;

rep=zeros(length(trainTestSetPathFeatures), length(tRep), 'like', tRep);     

                                                     
fprintf('Feature extraction  for %d vids: ', length(trainTestSetPathFeatures));
for i=1:length(trainTestSetPathFeatures)
    
    if mod(i,100)==0
        fprintf('%d ', i)
    end
    
    [desc, info, descParamUsed] = MediaName2Descriptor(trainTestSetPathFeatures{i}, descParam, pcaMap);
    
    load(trainTestSetPathFeatures{i});
    rep(i, :) = v;
    clear v


          
end

fprintf('\nDone!\n');


nEncoding=1;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(rep);
allDist{1}=temp * temp';

clear temp




all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


for k=1:nEncoding
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



acc1=mean(all_accuracy{1})

