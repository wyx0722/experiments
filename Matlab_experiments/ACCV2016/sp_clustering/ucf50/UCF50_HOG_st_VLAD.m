
global DATAopts;
DATAopts = UCFInit;
addpath('./../..')
addpath('./..')

% Parameter settings for descriptor extraction
clear descParam
descParam.Dataset='UCF50';
descParam.Func = @FEVid_IDT;
descParam.MediaType = 'IDT';
descParam.IDTfeature='HOG_iTraj';
descParam.Normalisation='ROOTSIFT'; % L2 or 'ROOTSIFT'
alpha=0.1;%for PN !!!!!!!change!!!!!!!

switch descParam.MediaType
    case 'IDT'
        if strfind(descParam.IDTfeature,'HOF')>0
            sizeDesc=108;   
        else%elseif  (strfind(descParam.IDTfeature,'HOG') + strfind(descParam.IDTfeature,'MBHx') + strfind(descParam.IDTfeature,'MBHy'))>0  
            sizeDesc=96;   
        end
        descParam.pcaDim = sizeDesc/2;
    case 'DeepF'
        descParam.pcaDim=256;%!!!
end

descParam.Clusters=[256 512];
descParam.spClusters=[32];


descParam

%%%%%%%%%%
bazePathFeatures='/home/ionut/asustor_ionut/Data/Features/UCF50/IDT/Videos/' %change


vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    if strfind(descParam.MediaType, 'DeepF')>0 
        vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '/' descParam.Layer '.txt'];
    elseif strfind(descParam.MediaType, 'IDT')>0 
        vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i))];
    end
end


                                           
%[vocabulary, pcaMap, st_d, skew, nElem, kurt] = CreateVocabularyKmeansPca_m(vocabularyImsPaths, descParam, ...
%                                                       descParam.numClusters, descParam.pcaDim);

[cell_Clusters, cell_spClusters, pcaMap] = CreateVocabularyKmeansPca_sptCl(vocabularyImsPaths, descParam);

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    if strfind(descParam.MediaType, 'DeepF')>0 
        pathFeatures{i}=[bazePathFeatures char(vids(i)) '/' descParam.Layer '.txt'];
    elseif strfind(descParam.MediaType, 'IDT')>0  
        pathFeatures{i}=[bazePathFeatures char(vids(i))];
    end
end



[tDesc info] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
    
t=VLAD_1_mean(tDesc, cell_Clusters{1}.vocabulary);
v256=zeros(length(vids), length(t), 'like', t); 

t=VLAD_1_mean(tDesc, cell_Clusters{2}.vocabulary);
v512=zeros(length(vids), length(t), 'like', t); 


t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{1}.vocabulary);
spV32=zeros(length(vids), length(t), 'like', t);

%parpool(13);
nDesc=zeros(1, length(allPathFeatures));
% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
for i=1:length(pathFeatures)
    %fprintf('%d \n', i)
    % Extract descriptors
    if mod(i,100) == 0
        fprintf('%d ', i);
    end
    
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
   % desc = NormalizeRowsUnit(desc);
    nDesc(i)=size(desc,1);
    
    v256(i, :) = VLAD_1_mean(desc, cell_Clusters{1}.vocabulary);
    v512(i, :) = VLAD_1_mean(desc, cell_Clusters{2}.vocabulary);
    
    
    spV32(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{1}.vocabulary);
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=3;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(PowerNormalization(v256, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v512, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV32, alpha));
allDist{3}=temp * temp';

clear temp


all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


for k=1:nEncoding

% 
% Leave-one-group-out cross-validation
for i=1:max(groups)
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

finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    finalAcc(j)=mean(mean(cat(2, all_accuracy{j}{:}), 2));
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));  
end


saveName2 = ['/home/ionut/asustor_ionut_2/Data/results/' 'mmm2016/ucf50/' 'videoRep/' DescParam2Name(descParam) '.mat']
save(saveName2, '-v7.3','v256','v512', 'spV32', 'descParam', 'nDesc');


saveName3 = ['/home/ionut/asustor_ionut_2/Data/results/' 'mmm2016/ucf50/' 'clsfOut/' DescParam2Name(descParam) '.mat']
save(saveName3, '-v7.3', 'all_clfsOut', 'all_accuracy', 'finalAcc', 'descParam');



