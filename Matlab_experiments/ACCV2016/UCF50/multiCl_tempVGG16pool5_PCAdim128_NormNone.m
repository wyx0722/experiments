
global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='TempVGG16Split1';
descParam.Normalisation='None'; %'ROOTSIFT';

descParam.pcaDim = 128;

descParam.orgClusters=512;
descParam.bovwCL=16;
descParam.smallCL=32;

descParam

%%%%%%%%%%
bazePathFeatures='/home/ionut/Data/action_temporal_vgg_16_split1_features_opticalFlow_tvL1_UCF50/Videos/' %change


vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '/pool5.txt'];
end

%[vocabulary, pcaMap, st_d, skew, nElem, kurt] = CreateVocabularyKmeansPca_m(vocabularyImsPaths, descParam, ...
%                                                       descParam.numClusters, descParam.pcaDim);

[pcaMap, orgCluster, bovwCluster, cell_smallCls] = CreateVocabularyKmeansPca_sepVocab(vocabularyImsPaths, descParam, descParam.orgClusters, descParam.bovwCL, descParam.smallCL, descParam.pcaDim);




%vocabulary = NormalizeRowsUnit(vocabulary); %make unit length

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i)) '/pool5.txt'];
end


[tDesc] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
    %tDesc=NormalizeRowsUnit(tDesc);
tVLAD=VLAD_1(tDesc, orgCluster.vocabulary);
tRep=getRepresentationMultiClusters(tDesc,bovwCluster, cell_smallCls, @VLAD_1);

vladNoMean=zeros(length(vids), length(tVLAD), 'like', tVLAD); 
maxPool=zeros(length(vids), length(tVLAD), 'like', tVLAD);

multiVLAD=zeros(length(vids), length(tRep), 'like', tRep);
multiMaxPool=zeros(length(vids), length(tRep), 'like', tRep);  

parpool(13);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
parfor i=1:length(pathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
   % desc = NormalizeRowsUnit(desc);
   
    
    vladNoMean(i, :)=VLAD_1(desc, orgCluster.vocabulary);
    maxPool(i, :)=maxPooling(desc, orgCluster.vocabulary);
    
    multiVLAD(i, :)=getRepresentationMultiClusters(desc,bovwCluster, cell_smallCls, @VLAD_1);
    multiMaxPool(i, :)=getRepresentationMultiClusters(desc,bovwCluster, cell_smallCls, @max_pooling);


         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=4;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(PowerNormalization(vladNoMean, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(maxPool, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(multiVLAD, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(multiMaxPool, alpha));
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
 

