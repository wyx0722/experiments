
global DATAopts;
DATAopts = UCF101Init;

addpath('./../..')
addpath('./..')

clear descParam
descParam.Dataset='UCF101';
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='SpVGG19';
descParam.Normalisation='None'; % L2 or 'ROOTSIFT'
alpha=0.5;%for PN !!!!!!!change!!!!!!!

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

%the baze path for features
bazePathFeatures='/home/ionut/asustor_ionut_2/Data/VGG_19_features_rawFrames_UCF101/Videos/'
descParam

[allVids, labs, splits] = GetVideosPlusLabels('Challenge');


%create the full path of the fetures for each video
allPathFeatures=cell(size(allVids));
for i=1:size(allVids, 1)
    
    if strfind(descParam.MediaType, 'DeepF')>0 
        allPathFeatures{i}=[bazePathFeatures allVids{i} '/' descParam.Layer '.txt'];
    elseif strfind(descParam.MediaType, 'IDT')>0 
        allPathFeatures{i}=[bazePathFeatures allVids{i}];
    end
end

% %get the data for a specific split
% trainTestSplit=splits(:, descParam.Split); %get the devision of date between training and testing set for the current split. Exclude the videos not included in the split (0 value)
% trainingSetPathFeatures=allPathFeatures(trainTestSplit==1); %get the trining set feature paths
% vocabularyPathFeatures=trainingSetPathFeatures(1:3:end); % build the vocabulary from a third of the videos of the training set
vocabularyPathFeatures=allPathFeatures(1:4:end);

%[pcaMap, orgCluster, bovwCluster, cell_smallCls, bigCluster] = CreateVocabularyKmeansPca_sepVocab(vocabularyPathFeatures, descParam, descParam.orgClusters, descParam.bovwCL, descParam.smallCL, descParam.pcaDim);
[cell_Clusters, cell_spClusters, pcaMap] = CreateVocabularyKmeansPca_sptCl(vocabularyPathFeatures, descParam);
 
                                            
                                            
[tDesc, info] = MediaName2Descriptor(allPathFeatures{1}, descParam, pcaMap);                                           

t=VLAD_1_mean(tDesc, cell_Clusters{1}.vocabulary);
v256=zeros(length(allPathFeatures), length(t), 'like', t); 

t=VLAD_1_mean(tDesc, cell_Clusters{2}.vocabulary);
v512=zeros(length(allPathFeatures), length(t), 'like', t); 

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
spV32=zeros(length(allPathFeatures), length(t), 'like', t);


nDesc=zeros(1, length(allPathFeatures));

fprintf('Feature extraction  for %d vids: ', length(allPathFeatures));

for i=1:length(allPathFeatures)
    %fprintf('%d \n', i)
    % Extract descriptors
    if mod(i,100) == 0
        fprintf('%d ', i);
    end
    
    
    [desc, info, descParamUsed] = MediaName2Descriptor(allPathFeatures{i}, descParam, pcaMap);
    nDesc(i)=size(desc,1);
    
    v256(i, :) = VLAD_1_mean(desc, cell_Clusters{1}.vocabulary);
    v512(i, :) = VLAD_1_mean(desc, cell_Clusters{2}.vocabulary);
    
    
    spV32(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
        
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

all_clfsOut=cell(nEncoding,3);
all_accuracy=cell(nEncoding,3);
mean_all_clfsOut=cell(nEncoding,1);
mean_all_accuracy=cell(nEncoding,1);

cRange = 100;
nReps = 1;
nFolds = 3;


for k=1:nEncoding
    k
    for i=1:3
        trainI=splits(:,i)==1;
        testI=~trainI;

        trainDist = allDist{k}(trainI, trainI);
        testDist = allDist{k}(testI, trainI);
        trainLabs = labs(trainI,:);
        testLabs = labs(testI, :);

        [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        accuracy = ClassificationAccuracy(clfsOut, testLabs);
        fprintf('accuracy: %.3f\n', mean(accuracy));

        all_clfsOut{k,i}=clfsOut;
        all_accuracy{k,i}=accuracy;
    end
end



finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    mean_all_clfsOut{j}=(all_clfsOut{j,1} + all_clfsOut{j,2} + all_clfsOut{j,3})./3;
    mean_all_accuracy{j}=(all_accuracy{j,1} + all_accuracy{j,2} + all_accuracy{j,3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end

saveName2 = ['/home/ionut/asustor_ionut_2/Data/results/' 'mmm2016/ucf101/' 'videoRep/' DescParam2Name(descParam) '.mat']
save(saveName2, '-v7.3','v256','v512', 'spV32', 'descParam', 'nDesc');


saveName3 = ['/home/ionut/asustor_ionut_2/Data/results/' 'mmm2016/ucf101/' 'clsfOut/' DescParam2Name(descParam) '.mat']
save(saveName3, '-v7.3', 'all_clfsOut', 'all_accuracy', 'finalAcc', 'descParam');


