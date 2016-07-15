
global DATAopts;
DATAopts = HMDB51Init;

addpath('./../..')
addpath('./..')

clear descParam
descParam.Dataset='HMBD51';
descParam.Func = @FEVid_IDT;
descParam.MediaType = 'IDT';
descParam.IDTfeature='HOG_iTraj';
descParam.Normalisation='ROOTSIFT'; % L2 or 'ROOTSIFT'

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


descParam.Clusters=[256 200 512];
descParam.spClusters=[10:10:100];

%the baze path for features
bazePathFeatures='/home/ionut/asustor_ionut_2/Data/iDT_Features_HMDB51/Videos/'
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

vocabularyPathFeatures=allPathFeatures(1:3:end);

%[pcaMap, orgCluster, bovwCluster, cell_smallCls] = CreateVocabularyKmeansPca_sepVocab(vocabularyPathFeatures, descParam, descParam.orgClusters, descParam.bovwCL, descParam.smallCL, descParam.pcaDim);
[cell_Clusters, cell_spClusters, pcaMap] = CreateVocabularyKmeansPca_sptCl(vocabularyPathFeatures, descParam);
                                           
                                            


[tDesc info] = MediaName2Descriptor(allPathFeatures{2}, descParam, pcaMap);
   
t=VLAD_1_mean(tDesc, cell_Clusters{1}.vocabulary);
v256=zeros(length(allPathFeatures), length(t), 'like', t); 

t=VLAD_1_mean(tDesc, cell_Clusters{2}.vocabulary);
v200=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean(tDesc, cell_Clusters{3}.vocabulary);
v512=zeros(length(allPathFeatures), length(t), 'like', t); 

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{1}.vocabulary);
spV10=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{2}.vocabulary);
spV20=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{3}.vocabulary);
spV30=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{4}.vocabulary);
spV40=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{5}.vocabulary);
spV50=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{6}.vocabulary);
spV60=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{7}.vocabulary);
spV70=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{8}.vocabulary);
spV80=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{9}.vocabulary);
spV90=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{10}.vocabulary);
spV100=zeros(length(allPathFeatures), length(t), 'like', t);


nDesc=zeros(1, length(allPathFeatures));

fprintf('Feature extraction  for %d vids: ', length(allPathFeatures));
parpool(3);
parfor i=1:length(allPathFeatures)
    fprintf('%d \n', i)
    
    [desc, info, descParamUsed] = MediaName2Descriptor(allPathFeatures{i}, descParam, pcaMap);
    
    nDesc(i)=size(desc,1);
     
    v256(i, :) = VLAD_1_mean(desc, cell_Clusters{1}.vocabulary);
    v200(i, :) = VLAD_1_mean(desc, cell_Clusters{2}.vocabulary);
    v512(i, :) = VLAD_1_mean(desc, cell_Clusters{3}.vocabulary);
    
    spV10(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{1}.vocabulary);
    spV20(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{2}.vocabulary);
    spV30(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{3}.vocabulary);
    spV40(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{4}.vocabulary);
    spV50(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{5}.vocabulary);
    spV60(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{6}.vocabulary);
    spV70(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{7}.vocabulary);
    spV80(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{8}.vocabulary);
    spV90(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{9}.vocabulary);
    spV100(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{10}.vocabulary);
    
     
   
        
     if i == 1
         descParamUsed
     end
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

%% Do classification
nEncoding=13;%!!!!!!!!change!!!!!!
allDist=cell(1, nEncoding);
alpha=0.5;

temp=NormalizeRowsUnit(PowerNormalization(v256, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v200, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v512, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV10, alpha));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV20, alpha));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV30, alpha));
allDist{6}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV40, alpha));
allDist{7}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV50, alpha));
allDist{8}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV60, alpha));
allDist{9}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV70, alpha));
allDist{10}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV80, alpha));
allDist{11}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV90, alpha));
allDist{12}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV100, alpha));
allDist{13}=temp * temp';


clear temp

%each row for the cell represents the results for all 3 splits
all_clfsOut=cell(nEncoding,3);
all_accuracy=cell(nEncoding,3);
mean_all_clfsOut=cell(nEncoding,1);
mean_all_accuracy=cell(nEncoding,1);

cRange = 100;
nReps = 1;
nFolds = 3;


parpool(7);
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

