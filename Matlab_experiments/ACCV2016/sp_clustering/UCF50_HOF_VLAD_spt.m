
global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_IDT;
descParam.MediaType = 'IDT';
descParam.IDTfeature='HOF_iTraj';
descParam.Normalisation='ROOTSIFT'; % L2 or 'ROOTSIFT'
descParam.info='spt';

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

[cell_Clusters, cell_spClusters, pcaMap] = CreateVocabularyKmeansPca_sptCl(vocabularyImsPaths, descParam);



%vocabulary = NormalizeRowsUnit(vocabulary); %make unit length

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i))];
end



[tDesc info] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
    
t=VLAD_1_mean(tDesc, cell_Clusters{1}.vocabulary);
v256=zeros(length(vids), length(t), 'like', t); 

t=VLAD_1_mean(tDesc, cell_Clusters{2}.vocabulary);
v512=zeros(length(vids), length(t), 'like', t); 

t=VLAD_1_mean_spClustering(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{1}.vocabulary);
spV8=zeros(length(vids), length(t), 'like', t);

t=VLAD_1_mean_spClustering(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{2}.vocabulary);
spV32=zeros(length(vids), length(t), 'like', t);

t=VLAD_1_mean_spClustering(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{3}.vocabulary);
spV64=zeros(length(vids), length(t), 'like', t);

t=VLAD_1_mean_spClustering(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{4}.vocabulary);
spV256=zeros(length(vids), length(t), 'like', t);


parpool(5);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
parfor i=1:length(pathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
   % desc = NormalizeRowsUnit(desc);
   
    
    v256(i, :) = VLAD_1_mean(desc, cell_Clusters{1}.vocabulary);
    v512(i, :) = VLAD_1_mean(desc, cell_Clusters{2}.vocabulary);
    
    spV8(i, :) = VLAD_1_mean_spClustering(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{1}.vocabulary);
    spV32(i, :) = VLAD_1_mean_spClustering(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{2}.vocabulary);
    spV64(i, :) = VLAD_1_mean_spClustering(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{3}.vocabulary);
    spV256(i, :) = VLAD_1_mean_spClustering(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{4}.vocabulary);
    
   
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=6;
allDist=cell(1, nEncoding);
alpha=0.5;

temp=NormalizeRowsUnit(PowerNormalization(v256, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v512, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV8, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV32, alpha));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV64, alpha));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV256, alpha));
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
 

