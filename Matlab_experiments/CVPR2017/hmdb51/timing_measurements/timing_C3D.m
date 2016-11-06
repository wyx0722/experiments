addpath('./../');%!!!!!!!
addpath('./../../');%!!!!!!!


global DATAopts;

datasetName='HMDB51';
mType='DeepF';
typeFeature=@FEVid_deepFeatures;
normStrategy='None';

%~~~~~~~~~~~~~~
layer='conv5b'
Net='C3D';
pathFeatures='/home/ionut/asustor_ionut/Data/mat_c3d_features_hmdb51/Videos/'%%%%%channge~~~~~~~~~~~~
%~~~~~~~~~~~~~~


descParam.Dataset=datasetName;
descParam.MediaType=mType;
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;
descParam.Layer=layer;
descParam.net=Net;
bazePathFeatures=pathFeatures
 
 descParam
 
 
if ~isempty(strfind(descParam.Dataset, 'HMDB51'))
    DATAopts = HMDB51Init;
    [allVids, labs, splits] = GetVideosPlusLabels();
elseif ~isempty(strfind(descParam.Dataset, 'UCF101'))
    DATAopts = UCF101Init;
    [allVids, labs, splits] = GetVideosPlusLabels('Challenge');
end

%create the full path of the fetures for each video
allPathFeatures=cell(size(allVids));
for i=1:size(allVids, 1)
    
    if ~isempty(strfind(descParam.MediaType, 'DeepF'))
        
        if ~isempty(strfind(descParam.net, 'C3D'))
            file_extension='.mat';
        else
            file_extension='.txt';
        end
        if ~isempty(strfind(descParam.Dataset, 'HMDB51'))    
            allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4) '/' descParam.Layer file_extension];
        end
        if ~isempty(strfind(descParam.Dataset, 'UCF101'))    
            allPathFeatures{i}=[bazePathFeatures allVids{i} '/' descParam.Layer file_extension];
        end
        
    elseif ~isempty(strfind(descParam.MediaType, 'IDT'))
        allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4)];
    elseif ~isempty(strfind(descParam.MediaType, 'Vid'))
        
        if ~isempty(strfind(descParam.Dataset, 'HMDB51'))
            allPathFeatures{i}=[DATAopts.videoPath, allVids{i}];
        else
            allPathFeatures{i}=sprintf(DATAopts.videoPath, allVids{i});
        end
    end
end

bazeVocab='/home/ionut/Data/UCF-101/VisualVocabulary/'
switch descParam.net
    case 'SpVGG19'
        name=[bazeVocab 'KmeansAndGmmFEVid_deepFeaturesClusters256DatasetUCF101Layerpool5MediaTypeDeepFNormalisationNonegmmSize256netSpVGG19pcaDim256_0_spClusters32_64_.mat']
    case 'C3D'
         name=[bazeVocab 'KmeansAndGmmFEVid_deepFeaturesClusters256DatasetUCF101Layerconv5bMediaTypeDeepFNormalisationNonegmmSize256netC3DpcaDim256_0_spClusters32_64_.mat']
    case 'TpVGG16'
        name=[bazeVocab 'KmeansAndGmmFEVid_deepFeaturesClusters256DatasetUCF101Layerpool5MediaTypeDeepFNormalisationNonegmmSize256netTempSplit1VGG16pcaDim256_0_spClusters32_64_.mat']       
end
load(name);



load('timing_permV.mat'); %permV=randperm(length(allPathFeatures));
subsetVideos=allPathFeatures(permV(1:500)); %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


[desc, info, descParam] = descParam.Func(subsetVideos{2}, descParam);


pca256_desc = desc * cell_pcaMap{1}.data.rot;
pca0_desc = desc * cell_pcaMap{2}.data.rot;


t=max_pooling_abs(pca256_desc, cell_Clusters{1}.vocabulary); VLMPFpca256=zeros(length(subsetVideos), length(t), 'like', t);
t=max_pooling_abs(pca0_desc, cell_Clusters{2}.vocabulary); VLMPFpca0=zeros(length(subsetVideos), length(t), 'like', t);

t=ST_VLMPF_abs(pca256_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); stVLMPFpca256=zeros(length(subsetVideos), length(t), 'like', t);
t=ST_VLMPF_abs(pca0_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);  stVLMPFpca0=zeros(length(subsetVideos), length(t), 'like', t);


t=VLAD_1(pca256_desc, cell_Clusters{1}.vocabulary); VLADpca256=zeros(length(subsetVideos), length(t), 'like', t);
t=VLAD_1(pca0_desc, cell_Clusters{2}.vocabulary); VLADpca0=zeros(length(subsetVideos), length(t), 'like', t);

t=mexFisherAssign(pca256_desc', cell_gmmModelName{1})'; FVpca256=zeros(length(subsetVideos), length(t), 'like', t);
t=mexFisherAssign(pca0_desc', cell_gmmModelName{2})'; FVpca0=zeros(length(subsetVideos), length(t), 'like', t);



t=max_pooling(pca256_desc, cell_Clusters{1}.vocabulary); VLMPFpca256_noabs=zeros(length(subsetVideos), length(t), 'like', t);
t=max_pooling(pca0_desc, cell_Clusters{2}.vocabulary); VLMPFpca0_noabs=zeros(length(subsetVideos), length(t), 'like', t);

t=ST_VLMPF(pca256_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); stVLMPFpca256_noabs=zeros(length(subsetVideos), length(t), 'like', t);
t=ST_VLMPF(pca0_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);  stVLMPFpca0_noabs=zeros(length(subsetVideos), length(t), 'like', t);



clear pca256_desc pca0_desc desc info


nDesc=zeros(1, length(subsetVideos));

t_VLMPFpca256 = zeros(1, length(subsetVideos));
t_VLMPFpca0 = zeros(1, length(subsetVideos));
t_stVLMPFpca256 = zeros(1, length(subsetVideos));
t_stVLMPFpca0 = zeros(1, length(subsetVideos));
t_VLADpca256 = zeros(1, length(subsetVideos));
t_VLADpca0 = zeros(1, length(subsetVideos));
t_FVpca256 = zeros(1, length(subsetVideos));
t_FVpca0 = zeros(1, length(subsetVideos));

t_VLMPFpca256_noabs = zeros(1, length(subsetVideos));
t_VLMPFpca0_noabs = zeros(1, length(subsetVideos));
t_stVLMPFpca256_noabs = zeros(1, length(subsetVideos));
t_stVLMPFpca0_noabs = zeros(1, length(subsetVideos));


fprintf('Feature extraction  for %d vids: ', length(subsetVideos));
for i=1:length(subsetVideos)
    if mod(i,100)==0
        fprintf('%d ', i)
    end
    
    [desc, info, descParamUsed] = descParam.Func(subsetVideos{i}, descParam);
    nDesc(i)=size(desc,1);
    
    
    pca256_desc = desc * cell_pcaMap{1}.data.rot;
    pca0_desc = desc * cell_pcaMap{2}.data.rot;
    
    tic
    VLMPFpca256(i, :)=max_pooling_abs(pca256_desc, cell_Clusters{1}.vocabulary);
    t_VLMPFpca256(i)=toc;
    
    tic
    VLMPFpca0(i, :)=max_pooling_abs(pca0_desc, cell_Clusters{2}.vocabulary);
    t_VLMPFpca0(i)=toc;
    
    tic
    stVLMPFpca256(i, :)=ST_VLMPF_abs(pca256_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
    t_stVLMPFpca256(i)=toc;
    
    tic
    stVLMPFpca0(i, :)=ST_VLMPF_abs(pca0_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
    t_stVLMPFpca0(i)=toc;
    
    tic
    VLADpca256(i, :)=VLAD_1(pca256_desc, cell_Clusters{1}.vocabulary);
    t_VLADpca256(i)=toc;
    
    tic
    VLADpca0(i, :)=VLAD_1(pca0_desc, cell_Clusters{2}.vocabulary);
    t_VLADpca0(i)=toc;
    
    tic
    FVpca256(i, :)=mexFisherAssign(pca256_desc', cell_gmmModelName{1})'; 
    t_FVpca256(i)=toc;
    
    tic
    FVpca0(i, :)=mexFisherAssign(pca0_desc', cell_gmmModelName{2})';
    t_FVpca0(i)=toc;
    
    
    
    tic
    VLMPFpca256_noabs(i, :)=max_pooling(pca256_desc, cell_Clusters{1}.vocabulary);
    t_VLMPFpca256_noabs(i)=toc;
    
    tic
    VLMPFpca0_noabs(i, :)=max_pooling(pca0_desc, cell_Clusters{2}.vocabulary);
    t_VLMPFpca0_noabs(i)=toc;
    
    tic
    stVLMPFpca256_noabs(i, :)=ST_VLMPF(pca256_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
    t_stVLMPFpca256_noabs(i)=toc;
    
    tic
    stVLMPFpca0_noabs(i, :)=ST_VLMPF(pca0_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
    t_stVLMPFpca0_noabs(i)=toc;
    
    
    
    
end
fprintf('\nDone!\n\n');

descParam

fprintf('\n\nSum of timing measurements for PCA256: \n')
fprintf('%.4f\n',sum(t_FVpca256));
fprintf('%.4f\n',sum(t_VLADpca256));
fprintf('%.4f\n',sum(t_VLMPFpca256));
fprintf('%.4f\n\n',sum(t_stVLMPFpca256));
fprintf('%.4f\n',sum(t_VLMPFpca256_noabs));
fprintf('%.4f\n\n',sum(t_stVLMPFpca256_noabs));

fprintf('\n\nSum of timing measurements for PCA0: \n')
fprintf('%.4f\n',sum(t_FVpca0));
fprintf('%.4f\n',sum(t_VLADpca0));
fprintf('%.4f\n',sum(t_VLMPFpca0));
fprintf('%.4f\n\n',sum(t_stVLMPFpca0));
fprintf('%.4f\n',sum(t_VLMPFpca0_noabs));
fprintf('%.4f\n',sum(t_stVLMPFpca0_noabs));

fprintf('\n\nTotal number of descriptors for %d videos: %d\n',length(subsetVideos), sum(nDesc));
