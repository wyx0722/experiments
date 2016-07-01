
global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_IDT;
descParam.MediaType = 'IDT';
descParam.IDTfeature='HOG_iTraj';
descParam.Normalisation='ROOTSIFT'; % L2 or 'ROOTSIFT'


if strfind(descParam.IDTfeature,'HOF')
    sizeDesc=108;
    
elseif  strfind(descParam.IDTfeature,'HOG') || strfind(descParam.IDTfeature,'MBHx') || strfind(descParam.IDTfeature,'MBHy')  
    sizeDesc=96;   
end

% pcaDim & vocabulary size
descParam.pcaDim = sizeDesc/2;

descParam.Clusters=[256 512];
descParam.spClusters=[8 32 64 256];


descParam

%%%%%%%%%%
bazePathFeatures='/home/ionut/asustor_ionut/Data/Features/UCF50/IDT/Videos/' %change


vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i))];
end


                                           
%[vocabulary, pcaMap, st_d, skew, nElem, kurt] = CreateVocabularyKmeansPca_m(vocabularyImsPaths, descParam, ...
%                                                       descParam.numClusters, descParam.pcaDim);

[cell_Clusters, cell_spClusters, pcaMap] = CreateVocabularyKmeansPca_spCl(vocabularyImsPaths, descParam);



%vocabulary = NormalizeRowsUnit(vocabulary); %make unit length

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i))];
end



[tDesc info] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
    
t=max_pooling(tDesc, cell_Clusters{1}.vocabulary);
v256=zeros(length(vids), length(t), 'like', t); 

t=max_pooling(tDesc, cell_Clusters{2}.vocabulary);
v512=zeros(length(vids), length(t), 'like', t); 

t=max_pooling_sp(tDesc,  cell_spClusters{1}.vocabulary, info.infoTraj(:, 8:9));
spV8=zeros(length(vids), length(t), 'like', t);

t=max_pooling_sp(tDesc,  cell_spClusters{2}.vocabulary, info.infoTraj(:, 8:9));
spV32=zeros(length(vids), length(t), 'like', t);

t=max_pooling_sp(tDesc,  cell_spClusters{3}.vocabulary, info.infoTraj(:, 8:9));
spV64=zeros(length(vids), length(t), 'like', t);

t=max_pooling_sp(tDesc,  cell_spClusters{4}.vocabulary, info.infoTraj(:, 8:9));
spV256=zeros(length(vids), length(t), 'like', t);


parpool(13);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
parfor i=1:length(pathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
   % desc = NormalizeRowsUnit(desc);
   
    
    v256(i, :) = max_pooling(desc, cell_Clusters{1}.vocabulary);
    v512(i, :) = max_pooling(desc, cell_Clusters{2}.vocabulary);
    
    spV8(i, :) =  max_pooling_sp(desc,  cell_spClusters{1}.vocabulary, info.infoTraj(:, 8:9));
    spV32(i, :) = max_pooling_sp(desc,  cell_spClusters{2}.vocabulary, info.infoTraj(:, 8:9));
    spV64(i, :) = max_pooling_sp(desc,  cell_spClusters{3}.vocabulary, info.infoTraj(:, 8:9));
    spV256(i, :) = max_pooling_sp(desc,  cell_spClusters{4}.vocabulary, info.infoTraj(:, 8:9));
    
   
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=6;
allDist=cell(1, nEncoding);
alpha=0.5;

temp=NormalizeRowsUnit(v256);
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(v512);
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(cat(2,v256, spV8));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(cat(2, v256, spV32));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(cat(2,v256, spV64));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(cat(2, v256, spV256));
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


% saveName = ['/home/ionut/Data/results/ICPR2016_rezults/' 'clfsOut/' 'encoding/'  DescParam2Name(descParam) '_vladNoMean_maxEncode_fisherVectors_.mat'];
% save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');
% 
%  saveName2 = ['/home/ionut/Data/results/ICPR2016_rezults/' 'videoRep/' 'encoding/' DescParam2Name(descParam) '_vladNoMean_maxEncode_fisherVectors_.mat'];
%  save(saveName2, '-v7.3', 'vladNoMean', 'maxEncode', 'fisherVectors');
%  
 

