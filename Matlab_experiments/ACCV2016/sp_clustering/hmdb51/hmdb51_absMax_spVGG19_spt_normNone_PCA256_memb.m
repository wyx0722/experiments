
global DATAopts;
DATAopts = HMDB51Init;

addpath('./../..')
addpath('./..')

clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='SpVGG19';
descParam.Normalisation='None'; % L2 or 'ROOTSIFT'

switch descParam.MediaType
    case 'IDT'
        if strfind(descParam.IDTfeature,'HOF')>0
            sizeDesc=108;   
        elseif  strfind(descParam.IDTfeature,'HOG')>0 || strfind(descParam.IDTfeature,'MBHx')>0 || strfind(descParam.IDTfeature,'MBHy')>0  
            sizeDesc=96;   
        end
        descParam.pcaDim = sizeDesc/2;
    case 'DeepF'
        descParam.pcaDim=256;%!!!
end


descParam.Clusters=[256 320 512];
descParam.spClusters=[8 32 64 256];


descParam.Dataset='HMBD51Split1';



[allVids, labs, splits] = GetVideosPlusLabels();

%the baze path for features
bazePathFeatures='/home/ionut/asustor_ionut_2/Data/hmdb51_VGG_19_features_rawFrames/Videos/'
descParam

%create the full path of the fetures for each video
allPathFeatures=cell(size(allVids));
for i=1:size(allVids, 1)
    
    if strfind(descParam.MediaType, 'DeepF')>0 
        allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4) '/' descParam.Layer '.txt'];
    elseif strfind(descParam.MediaType, 'IDT')>0 
        allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4)];
    end
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


%[vocabulary, pcaMap, st_d, skew, nElem, kurt] = CreateVocabularyKmeansPca_m(vocabularyPathFeatures, descParam, ...
%                                                descParam.numClusters, descParam.pcaDim); 



%[pcaMap, orgCluster, bovwCluster, cell_smallCls] = CreateVocabularyKmeansPca_sepVocab(vocabularyPathFeatures, descParam, descParam.orgClusters, descParam.bovwCL, descParam.smallCL, descParam.pcaDim);
[cell_Clusters, cell_spClusters, pcaMap] = CreateVocabularyKmeansPca_sptCl(vocabularyPathFeatures, descParam);
                                           
                                            


[tDesc info] = MediaName2Descriptor(trainTestSetPathFeatures{1}, descParam, pcaMap);
    
t=abs_max_pooling(tDesc, cell_Clusters{1}.vocabulary);
m256=zeros(length(trainTestSetPathFeatures), length(t), 'like', t); 

t=abs_max_pooling(tDesc, cell_Clusters{2}.vocabulary);
m320=zeros(length(trainTestSetPathFeatures), length(t), 'like', t);

t=abs_max_pooling(tDesc, cell_Clusters{3}.vocabulary);
m512=zeros(length(trainTestSetPathFeatures), length(t), 'like', t); 

t=abs_maxPooling_memb(tDesc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
spM8=zeros(length(trainTestSetPathFeatures), length(t), 'like', t);

t=abs_maxPooling_memb(tDesc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary);
spM32=zeros(length(trainTestSetPathFeatures), length(t), 'like', t);

t=abs_maxPooling_memb(tDesc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
spM64=zeros(length(trainTestSetPathFeatures), length(t), 'like', t);

t=abs_maxPooling_memb(tDesc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{4}.vocabulary);
spM256=zeros(length(trainTestSetPathFeatures), length(t), 'like', t);


nDesc=zeros(1, length(trainTestSetPathFeatures));

fprintf('Feature extraction  for %d vids: ', length(trainTestSetPathFeatures));
parpool(2);
parfor i=1:length(trainTestSetPathFeatures)
    fprintf('%d \n', i)
    
    [desc, info, descParamUsed] = MediaName2Descriptor(trainTestSetPathFeatures{i}, descParam, pcaMap);
    
    nDesc(i)=size(desc,1);
     
    m256(i, :) = abs_max_pooling(desc, cell_Clusters{1}.vocabulary);
    m320(i, :) = abs_max_pooling(desc, cell_Clusters{2}.vocabulary);
    m512(i, :) = abs_max_pooling(desc, cell_Clusters{3}.vocabulary);
    
    spM8(i, :) = abs_maxPooling_memb(desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
    spM32(i, :) = abs_maxPooling_memb(desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary);
    spM64(i, :) = abs_maxPooling_memb(desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    spM256(i, :) = abs_maxPooling_memb(desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{4}.vocabulary);
    
     
   
        
     if i == 1
         descParamUsed
     end
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

%% Do classification
nEncoding=7;
allDist=cell(1, nEncoding);
alpha=0.5;

temp=NormalizeRowsUnit(PowerNormalization(m256, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(m320, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(m512, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spM8, alpha));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spM32, alpha));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spM64, alpha));
allDist{6}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spM256, alpha));
allDist{7}=temp * temp';

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


% acc1=mean(all_accuracy{1})
% acc2=mean(all_accuracy{2})
% acc3=mean(all_accuracy{3})
% acc4=mean(all_accuracy{4})
% acc5=mean(all_accuracy{5})
% acc6=mean(all_accuracy{6})
% acc7=mean(all_accuracy{7})
% acc8=mean(all_accuracy{8})



% saveName2 = ['/home/ionut/Data/results/' 'accv2016/' DescParam2Name(descParam) '.mat']
% save(saveName2, '-v7.3', 'vladNoMean', 'stdDiff', 'multiVLAD', 'multiStdDiff');
%  
