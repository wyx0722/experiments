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
%pathFeatures='/media/HDS2-UTX/ionut/Data/conv5b_pool5_mat_c3d_features_hmdb51/Videos/'%%%%%channge~~~~~~~~~~~~
pathFeatures='/home/ionut/asustor_ionut/Data/mat_c3d_features_hmdb51/Videos/'
%~~~~~~~~~~~~~~


d=0;


alpha=0.5;
nPar=5;


descParam.Details='__SecondTrial__';%!!!!!

descParam.Dataset=datasetName;
descParam.MediaType=mType;
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;

descParam.Clusters=[64 128 256 512];
descParam.spClusters=[2 4 8 16 32 64 128 256];


switch descParam.MediaType
    
    case 'Vid'
        descParam.NumBlocks = [3 3 2];
    
        if exist('fsr', 'var')
            descParam.FrameSampleRate = fsr;
            descParam.BlockSize = [8 8 6/fsr];
        else
            descParam.BlockSize = [8 8 6];
        end

        descParam.NumOr = 8;
        sRow = [1 3];
        sCol = [1 1];
        
        sizeDesc=144;
        if exist('d', 'var')
            descParam.pcaDim=d;
        else
            descParam.pcaDim=sizeDesc/2;
        end
        
    case 'IDT'
        descParam.IDTfeature=iDTfeature;
        
        if ~isempty(strfind(descParam.IDTfeature,'HOF'))
            sizeDesc=108;   
        else%if  strfind(descParam.IDTfeature,'HOG')>0 || strfind(descParam.IDTfeature,'MBHx')>0 || strfind(descParam.IDTfeature,'MBHy')>0  
            sizeDesc=96;   
        end
        if exist('d', 'var')
            descParam.pcaDim=d;
        else
            descParam.pcaDim=sizeDesc/2;
        end
        bazePathFeatures=pathFeatures
        
    case 'DeepF'
        descParam.Layer=layer;
        descParam.net=Net;
 
        sizeDesc=512;
        if exist('d', 'var')
            descParam.pcaDim=d;
        else
            descParam.pcaDim=sizeDesc/2;
        end
        bazePathFeatures=pathFeatures
end


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
            
        allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4) '/' descParam.Layer file_extension];
        
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

vocabularyPathFeatures=allPathFeatures(1:4:end);%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[cell_Clusters, cell_spClusters, pcaMap] = CreateVocabularyKmeansPca_sptCl(vocabularyPathFeatures, descParam);
 



[desc, info, descParam] = descParam.Func(allPathFeatures{1}, descParam);

pca_desc = desc * pcaMap.data.rot;

t=ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); sp2cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); sp4cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); sp8cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{4}.vocabulary); sp16cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary); sp32cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{6}.vocabulary); sp64cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{7}.vocabulary); sp128cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{8}.vocabulary); sp256cl256=zeros(length(allPathFeatures), length(t), 'like', t);

t=ST_VLMPF_abs(pca_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary); sp32cl64=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary); sp32cl128=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca_desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary); sp32cl512=zeros(length(allPathFeatures), length(t), 'like', t);


t=VLAD_1(pca_desc, cell_Clusters{3}.vocabulary); vlad256=zeros(length(allPathFeatures), length(t), 'like', t);


nDesc=zeros(1, length(allPathFeatures));

%parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(allPathFeatures));
for i=1:length(allPathFeatures)%parfor i=1:length(allPathFeatures)
    if mod(i, 100)==0
        fprintf('%d ', i)%fprintf('%d \n', i)
    end
    % Extract descriptors
    
    [desc, info, descParamUsed] = descParam.Func(allPathFeatures{i}, descParam);
    nDesc(i)=size(desc,1);
    
    pca_desc = desc * pcaMap.data.rot;


    sp2cl256(i, :) =ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); 
    sp4cl256(i, :) =ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); 
    sp8cl256(i, :) =ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); 
    sp16cl256(i, :) =ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{4}.vocabulary); 
    sp32cl256(i, :) =ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary); 
    sp64cl256(i, :) =ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{6}.vocabulary); 
    sp128cl256(i, :) =ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{7}.vocabulary); 
    sp256cl256(i, :) =ST_VLMPF_abs(pca_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{8}.vocabulary);

    sp32cl64(i, :) =ST_VLMPF_abs(pca_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary); 
    sp32cl128(i, :) =ST_VLMPF_abs(pca_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary); 
    sp32cl512(i, :) =ST_VLMPF_abs(pca_desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary);

    vlad256(i, :)=VLAD_1(pca_desc, cell_Clusters{3}.vocabulary);
    
    
         if i == 1
             descParamUsed
         end
         
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

nEncoding=13;
allDist=cell(1, nEncoding);


t_feature=sp2cl256;
t_feature(:, end-(size(cell_spClusters{1}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{1}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{1}=temp * temp';

t_feature=sp4cl256;
t_feature(:, end-(size(cell_spClusters{2}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{2}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{2}=temp * temp';

t_feature=sp8cl256;
t_feature(:, end-(size(cell_spClusters{3}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{3}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{3}=temp * temp';

t_feature=sp16cl256;
t_feature(:, end-(size(cell_spClusters{4}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{4}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{4}=temp * temp';

t_feature=sp32cl256;
t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{5}=temp * temp';

t_feature=sp64cl256;
t_feature(:, end-(size(cell_spClusters{6}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{6}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{6}=temp * temp';

t_feature=sp128cl256;
t_feature(:, end-(size(cell_spClusters{7}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{7}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{7}=temp * temp';

t_feature=sp256cl256;
t_feature(:, end-(size(cell_spClusters{8}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{8}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{8}=temp * temp';

    
t_feature=sp32cl64;
t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{1}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{1}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{9}=temp * temp';

t_feature=sp32cl128;
t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{2}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{2}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{10}=temp * temp';

t_feature=sp32cl512;
t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{4}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{4}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{11}=temp * temp';

clear t_feature

temp=NormalizeRowsUnit(PowerNormalization(vlad256, alpha)); allDist{12}=temp * temp';

temp=NormalizeRowsUnit(sp32cl256(:, 1:size(cell_Clusters{3}.vocabulary,1)*size(cell_Clusters{3}.vocabulary,2)));
allDist{13}=temp * temp';

clear temp

%each row for the cell represents the results for all 3 splits
all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);
clfsOut=cell(1,nEncoding);
accuracy=cell(1,nEncoding);
%mean_all_clfsOut=cell(nEncoding,1);
mean_all_accuracy=cell(nEncoding,1);

cRange = 100;
nReps = 1;
nFolds = 3;



parpool(3);
%%
for k=1:nEncoding
    k
    parfor i=1:3
        
        trainI = splits(:,i) == 1;
        
       if ~isempty(strfind(datasetName, 'HMDB51'))
            testI  = splits(:,i) == 2;
       elseif ~isempty(strfind(datasetName, 'UCF101'))
            testI=~trainI;
       end
       
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        trainDist = allDist{k}(trainI, trainI);
        testDist = allDist{k}(testI, trainI);
        

        [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
        %fprintf('accuracy: %.3f\n', accuracy);
    end
     all_clfsOut{k}=clfsOut;
     all_accuracy{k}=accuracy;
     fprintf('Accuracy for encoding %d: %.3f\n',k, mean((all_accuracy{k}{1} + all_accuracy{k}{2} + all_accuracy{k}{3})./3));
end

delete(gcp('nocreate')) %///
%%%%

clear allDist


finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('%.3f\n', finalAcc(j));

    
end


 bazeSavePath='/home/ionut/asustor_ionut/Data/results/cvpr2017/hmdb51/clusters/secondTrial/';
 
 
saveName = [bazeSavePath 'clfsOut/'  DescParam2Name(descParam) '.mat']
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy', 'mean_all_accuracy');

saveName2 = [bazeSavePath 'videoRep/'  DescParam2Name(descParam) '.mat']
save(saveName2, '-v7.3', 'descParam', 'sp2cl256', 'sp4cl256', 'sp8cl256', 'sp16cl256', 'sp32cl256', 'sp64cl256', 'sp128cl256', 'sp256cl256', 'sp32cl64', 'sp32cl128', 'sp32cl512', 'vlad256');



 clfsOut_sp32cl256 = all_clfsOut(5);
 acc_sp32cl256 = all_accuracy(5);
 
 intructions_compute_acc='mean((acc_sp32cl256{1}{1} + acc_sp32cl256{1}{2} + acc_sp32cl256{1}{3})./3)';
 
saveName = [bazeSavePath 'clfsOut/'  DescParam2Name(descParam) '_PNL2__sp32cl256.mat']
save(saveName, '-v7.3', 'descParam', 'clfsOut_sp32cl256', 'acc_sp32cl256', 'intructions_compute_acc');

saveName2 = [bazeSavePath 'videoRep/'  DescParam2Name(descParam) '__sp32cl256.mat']
save(saveName2, '-v7.3', 'descParam', 'sp32cl256');


