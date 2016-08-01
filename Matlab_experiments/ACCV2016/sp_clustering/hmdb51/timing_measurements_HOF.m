
global DATAopts;
DATAopts = HMDB51Init;

addpath('./../..')
addpath('./..')

clear descParam
descParam.Dataset='HMBD51';
descParam.Func = @FEVid_IDT;
descParam.MediaType = 'IDT';
descParam.IDTfeature='HOF_iTraj';
descParam.Normalisation='ROOTSIFT'; % L2 or 'ROOTSIFT'
alpha=0.1;%for PN !!!!!!!change!!!!!!!



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
   

t=VLAD_1_mean(tDesc, cell_Clusters{3}.vocabulary);
v256=zeros(length(allPathFeatures), length(t), 'like', t); 
v256_fast=v256;

t=VLAD_1_mean(tDesc, cell_Clusters{4}.vocabulary);
v512=zeros(length(allPathFeatures), length(t), 'like', t); 
v512_fast=v512;



t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{3}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{5}.vocabulary);
spV32=zeros(length(allPathFeatures), length(t), 'like', t);
spV32_fast=spV32;


load('timing_permV.mat'); %permV=randperm(length(allPathFeatures));
randVideos=allPathFeatures(permV(1:500));


nDesc=zeros(1, length(randVideos));


tDescExtr=zeros(1, length(randVideos));
nFrames=zeros(1, length(randVideos));
t_v256=zeros(1, length(randVideos));
t_v256_fast=zeros(1, length(randVideos));
t_v512=zeros(1, length(randVideos));
t_v512_fast=zeros(1, length(randVideos));
t_spV32=zeros(1, length(randVideos));
t_spV32_fast=zeros(1, length(randVideos));

fprintf('Feature extraction  for %d vids: ', length(randVideos));
%parpool(2);
for i=1:length(randVideos)
    if mod(i,10)==0
        fprintf('%d ', i)
    end
    
    tic
    [desc, info, descParamUsed] = MediaName2Descriptor(randVideos{i}, descParam, pcaMap);
    tDescExtr(i)=toc;
    
    nDesc(i)=size(desc,1);
    nFrames(i)=max(info.infoTraj(:, 1));
    
    tic
    v256(i, :) = VLAD_1_mean(desc, cell_Clusters{3}.vocabulary);
    t_v256(i)=toc;
    
    tic
    v256_fast(i, :) = fast_VLAD_1_mean(desc, cell_Clusters{3}.vocabulary);
    t_v256_fast(i)=toc;    
    
    
    
    tic
    v512(i, :) = VLAD_1_mean(desc, cell_Clusters{4}.vocabulary);
    t_v512(i)=toc;
    
    tic
    v512_fast(i, :) = fast_VLAD_1_mean(desc, cell_Clusters{4}.vocabulary);
    t_v512_fast(i)=toc;
    
    tic
    spV32(i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{3}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{5}.vocabulary);
    t_spV32(i)=toc;
    
    tic
    spV32_fast(i, :) = fast_VLAD_1_mean_spClustering_memb(desc, cell_Clusters{3}.vocabulary, info.infoTraj(:, 8:10), cell_spClusters{5}.vocabulary);
    t_spV32_fast(i)=toc;
    
      
        
     if i == 1
         descParamUsed
     end
end
%delete(gcp('nocreate'))
fprintf('\nDone!\n');


fprintf('Average time for %d videos VLAD256: %.3f \n', length(randVideos), mean(t_v256));
fprintf('Average time for %d videos VLAD256 *fast*: %.3f \n \n', length(randVideos), mean(t_v256_fast));
fprintf('Average time for %d videos VLAD512: %.3f \n', length(randVideos), mean(t_v512));
fprintf('Average time for %d videos VLAD512 *fast*: %.3f \n \n', length(randVideos), mean(t_v512_fast));
fprintf('Average time for %d videos st32: %.3f \n', length(randVideos), mean(t_spV32));
fprintf('Average time for %d videos st32 *fast*: %.3f \n \n', length(randVideos), mean(t_spV32_fast));

name=['./rezTiming/' DescParam2Name(descParam) '.mat']
save(name, '-v7.3', 'tDescExtr', 'nDesc', 'nFrames', 't_v256', 't_v256_fast', 't_v512', 't_v512_fast', 't_spV32', 't_spV32_fast');