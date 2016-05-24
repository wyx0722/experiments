function [all_accuracy, all_clfsOut ]=ucf101Framework(descParam, pathFeat, nPar )

global DATAopts;
DATAopts = UCF101Init;



[allVids, labs, splits] = GetVideosPlusLabels('Challenge');

%the baze path for features
bazePathFeatures=pathFeat

%create the full path of the fetures for each video
allPathFeatures=cell(size(allVids));
for i=1:size(allVids, 1)
    
    if strfind(descParam.MediaType, 'DeepF')>0 
        allPathFeatures{i}=[bazePathFeatures allVids{i} '/' descParam.Layer '.txt'];
    elseif strfind(descParam.MediaType, 'IDT')>0 
        allPathFeatures{i}=[bazePathFeatures allVids{i}];
    end
end

%get the data for a specific split
trainTestSplit=splits(:, descParam.Split); %get the devision of date between training and testing set for the current split. Exclude the videos not included in the split (0 value)
trainingSetPathFeatures=allPathFeatures(trainTestSplit==1); %get the trining set feature paths
vocabularyPathFeatures=trainingSetPathFeatures(1:3:end); % build the vocabulary from a third of the videos of the training set


[pcaMap, orgCluster, bovwCluster, cell_smallCls, bigCluster] = CreateVocabularyKmeansPca_sepVocab(vocabularyPathFeatures, descParam, descParam.orgClusters, descParam.bovwCL, descParam.smallCL, descParam.pcaDim);

                                            
                                            
[tDesc] = MediaName2Descriptor(allPathFeatures{1}, descParam, pcaMap);                                           

tVLAD=VLAD_1(tDesc, orgCluster.vocabulary);
tRep=getRepresentationMultiClusters(tDesc,bovwCluster, cell_smallCls, @VLAD_1);

vladNoMean=zeros(length(allPathFeatures), length(tVLAD), 'like', tVLAD); 
stdDiff=zeros(length(allPathFeatures), length(tVLAD), 'like', tVLAD);

multiVLAD=zeros(length(allPathFeatures), length(tRep), 'like', tRep);
multiStdDiff=zeros(length(allPathFeatures), length(tRep), 'like', tRep);  

tVLAD512=VLAD_1(tDesc, bigCluster.vocabulary);
vladNoMean512=zeros(length(allPathFeatures), length(tVLAD512), 'like', tVLAD512); 

fprintf('Feature extraction  for %d vids: ', length(allPathFeatures));
parpool(nPar);
parfor i=1:length(allPathFeatures)
    fprintf('%d \n', i)
    
    [desc, info, descParamUsed] = MediaName2Descriptor(allPathFeatures{i}, descParam, pcaMap);
    
    vladNoMean(i, :)=VLAD_1(desc, orgCluster.vocabulary);
    stdDiff(i, :)=stdDiff_VLAD(desc, orgCluster.vocabulary, orgCluster.st_d);
    
    multiVLAD(i, :)=getRepresentationMultiClusters(desc,bovwCluster, cell_smallCls, @VLAD_1);
    multiStdDiff(i, :)=getRepresentationMultiClusters(desc,bovwCluster, cell_smallCls, @stdDiff_VLAD);
    
    vladNoMean512(i, :)=VLAD_1(desc, bigCluster.vocabulary);
        
     if i == 1
         descParamUsed
     end
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

%% Do classification

initDim=size(orgCluster.vocabulary,2);
alpha=0.5;

intraL2_vladNoMean = intranormalizationFeatures( vladNoMean, initDim );
intraL2_stdDiff = intranormalizationFeatures( stdDiff, initDim );
intraL2_multiVLAD = intranormalizationFeatures( multiVLAD, initDim );
intraL2_multiStdDiff = intranormalizationFeatures( multiStdDiff, initDim );
intraL2_vladNoMean512= intranormalizationFeatures( vladNoMean512, initDim );

nEncoding=9;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(PowerNormalization(intraL2_vladNoMean, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraL2_stdDiff, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraL2_multiVLAD, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraL2_multiStdDiff, alpha));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(cat(2,intraL2_vladNoMean,intraL2_stdDiff ), alpha));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(cat(2,intraL2_multiVLAD,intraL2_multiStdDiff ), alpha));
allDist{6}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(cat(2,intraL2_vladNoMean, intraL2_multiVLAD ), alpha));
allDist{7}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(cat(2,intraL2_vladNoMean,intraL2_stdDiff, intraL2_multiVLAD, intraL2_multiStdDiff), alpha));
allDist{8}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intraL2_vladNoMean512, alpha));
allDist{9}=temp * temp';

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
end

delete(gcp('nocreate'))

acc1=mean(all_accuracy{1})
acc2=mean(all_accuracy{2})
acc3=mean(all_accuracy{3})
acc4=mean(all_accuracy{4})
acc5=mean(all_accuracy{5})
acc6=mean(all_accuracy{6})
acc7=mean(all_accuracy{7})
acc8=mean(all_accuracy{8})
acc9=mean(all_accuracy{9})


saveName2 = ['/home/ionut/Data/results/' 'accv2016/' 'ucf101/' DescParam2Name(descParam) '.mat']
save(saveName2, '-v7.3', 'vladNoMean', 'stdDiff', 'multiVLAD', 'multiStdDiff', 'vladNoMean512');
 

end
