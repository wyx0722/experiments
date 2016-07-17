
global DATAopts;
DATAopts = HMDB51Init;

addpath('./../..')
addpath('./..')

clear descParam
descParam.Dataset='HMBD51_t2__';
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='TempSplit1VGG16';
descParam.Normalisation='None'; % L2 or 'ROOTSIFT'
alpha=0.5;%for PN !!!!!!!change!!!!!!!



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


descParam.Clusters=[64 128 256 512];
descParam.spClusters=[2     4     8    16    32    64   128   256];

%the baze path for features
bazePathFeatures='/home/ionut/asustor_ionut_2/Data/hmdb51_action_temporal_vgg_16_split1_features_opticalFlow_tvL1/Videos/'
descParam





[allVids, labs, splits] = GetVideosPlusLabels();



%create the full path of the fetures for each video
allPathFeatures=cell(size(allVids));
for i=1:size(allVids, 1)
    
    if strfind(descParam.MediaType, 'DeepF')>0 
        allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4) '/' descParam.Layer '.txt'];
    elseif strfind(descParam.MediaType, 'IDT')>0 
        allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4)];
    end
end




%[vocabulary, pcaMap, st_d, skew, nElem, kurt] = CreateVocabularyKmeansPca_m(vocabularyPathFeatures, descParam, ...
%                                                descParam.numClusters, descParam.pcaDim); 

vocabularyPathFeatures=allPathFeatures(1:5:end);%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%[pcaMap, orgCluster, bovwCluster, cell_smallCls] = CreateVocabularyKmeansPca_sepVocab(vocabularyPathFeatures, descParam, descParam.orgClusters, descParam.bovwCL, descParam.smallCL, descParam.pcaDim);
[cell_Clusters, cell_spClusters, pcaMap] = CreateVocabularyKmeansPca_sptCl(vocabularyPathFeatures, descParam);
                                           
                                            


[tDesc info] = MediaName2Descriptor(allPathFeatures{2}, descParam, pcaMap);
   
t=VLAD_1_mean(tDesc, cell_Clusters{1}.vocabulary);
v64=zeros(length(allPathFeatures), length(t), 'like', t); 

t=VLAD_1_mean(tDesc, cell_Clusters{2}.vocabulary);
v128=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean(tDesc, cell_Clusters{3}.vocabulary);
v256=zeros(length(allPathFeatures), length(t), 'like', t); 

t=VLAD_1_mean(tDesc, cell_Clusters{4}.vocabulary);
v512=zeros(length(allPathFeatures), length(t), 'like', t); 


t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
spV2=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary);
spV4=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
spV8=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{4}.vocabulary);
spV16=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary);
spV32=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{6}.vocabulary);
spV64=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{7}.vocabulary);
spV128=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{8}.vocabulary);
spV256=zeros(length(allPathFeatures), length(t), 'like', t);



nDesc=zeros(1, length(allPathFeatures));

fprintf('Feature extraction  for %d vids: ', length(allPathFeatures));
parpool(8);
parfor i=1:length(allPathFeatures)
    fprintf('%d \n', i)
    
    [desc, info, descParamUsed] = MediaName2Descriptor(allPathFeatures{i}, descParam, pcaMap);
    
    nDesc(i)=size(desc,1);
     
    v64(i, :) = VLAD_1_mean(desc, cell_Clusters{1}.vocabulary);
    v128(i, :) = VLAD_1_mean(desc, cell_Clusters{2}.vocabulary);
    v256(i, :) = VLAD_1_mean(desc, cell_Clusters{3}.vocabulary);
    v512(i, :) = VLAD_1_mean(desc, cell_Clusters{4}.vocabulary);
    
    spV2(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
    spV4(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary);
    spV8(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    spV16(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{4}.vocabulary);
    spV32(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary);
    spV64(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{6}.vocabulary);
    spV128(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{7}.vocabulary);
    spV256(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{8}.vocabulary);

    
     
   
        
     if i == 1
         descParamUsed
     end
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

%% Do classification
nEncoding=12;%!!!!!!!!change!!!!!!
allDist=cell(1, nEncoding);


temp=NormalizeRowsUnit(PowerNormalization(v64, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v128, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v256, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v512, alpha));
allDist{4}=temp * temp';


temp=NormalizeRowsUnit(PowerNormalization(spV2, alpha));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV4, alpha));
allDist{6}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV8, alpha));
allDist{7}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV16, alpha));
allDist{8}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV32, alpha));
allDist{9}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV64, alpha));
allDist{10}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV128, alpha));
allDist{11}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV256, alpha));
allDist{12}=temp * temp';



clear temp

%each row for the cell represents the results for all 3 splits
all_clfsOut=cell(nEncoding,3);
all_accuracy=cell(nEncoding,3);
mean_all_clfsOut=cell(nEncoding,1);
mean_all_accuracy=cell(nEncoding,1);

cRange = 100;
nReps = 1;
nFolds = 3;


parpool(6);
parfor k=1:nEncoding
    k
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI  = splits(:,i) == 2;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        trainDist = allDist{k}(trainI, trainI);
        testDist = allDist{k}(testI, trainI);
        

        [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        accuracy = ClassificationAccuracy(clfsOut, testLabs);
        fprintf('accuracy: %.3f\n', accuracy);

        all_clfsOut{k,i}=clfsOut;
        all_accuracy{k,i}=accuracy;
    end
end

delete(gcp('nocreate'))


finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    mean_all_clfsOut{j}=(all_clfsOut{j,1} + all_clfsOut{j,2} + all_clfsOut{j,3})./3;
    mean_all_accuracy{j}=(all_accuracy{j,1} + all_accuracy{j,2} + all_accuracy{j,3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end

saveName2 = ['/home/ionut/asustor_ionut_2/Data/results/' 'mmm2016/hmdb51/' 'videoRep/' DescParam2Name(descParam) '.mat']
save(saveName2, '-v7.3', 'v64', 'v128','v256','v512', 'spV2', 'spV4', 'spV8', 'spV16', 'spV32','spV64','spV128','spV256','descParam', 'nDesc');


saveName3 = ['/home/ionut/asustor_ionut_2/Data/results/' 'mmm2016/hmdb51/' 'clsfOut/' DescParam2Name(descParam) '.mat']
save(saveName3, '-v7.3', 'all_clfsOut', 'all_accuracy', 'mean_all_clfsOut', 'mean_all_accuracy', 'finalAcc', 'descParam');

