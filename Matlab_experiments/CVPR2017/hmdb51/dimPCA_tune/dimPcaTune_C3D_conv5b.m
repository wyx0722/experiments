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
pathFeatures='/media/HDS2-UTX/ionut/Data/conv5b_pool5_mat_c3d_features_hmdb51/Videos/'%%%%%channge~~~~~~~~~~~~
%~~~~~~~~~~~~~~


d=[64 128 256 0];

alpha=0.5;
nPar=5;



descParam.Dataset=datasetName;
descParam.MediaType=mType;
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;

descParam.Clusters=[256];
descParam.spClusters=[32];


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

[cell_Clusters, cell_spClusters, cell_pcaMap] = CreateVocabularyKmeansPca_sptCl_multiplePCA(vocabularyPathFeatures, descParam);
 



[desc, info, descParam] = descParam.Func(allPathFeatures{1}, descParam);

pca64_desc = desc * cell_pcaMap{1}.data.rot;
pca128_desc = desc * cell_pcaMap{2}.data.rot;
pca256_desc = desc * cell_pcaMap{3}.data.rot;
pca0_desc = desc * cell_pcaMap{4}.data.rot;



t=ST_VLMPF_abs(pca64_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); sp32cl256pca64=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca128_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); sp32cl256pca128=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca256_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); sp32cl256pca256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca0_desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); sp32cl256pca0=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1(pca256_desc, cell_Clusters{3}.vocabulary); v256pca256=zeros(length(allPathFeatures), length(t), 'like', t);
t=VLAD_1(pca0_desc, cell_Clusters{4}.vocabulary); v256pca0=zeros(length(allPathFeatures), length(t), 'like', t);


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
    
    pca64_desc = desc * cell_pcaMap{1}.data.rot;
    pca128_desc = desc * cell_pcaMap{2}.data.rot;
    pca256_desc = desc * cell_pcaMap{3}.data.rot;
    pca0_desc = desc * cell_pcaMap{4}.data.rot;

 
    sp32cl256pca64(i, :)=ST_VLMPF_abs(pca64_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); 
    sp32cl256pca128(i, :)=ST_VLMPF_abs(pca128_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); 
    sp32cl256pca256(i, :)=ST_VLMPF_abs(pca256_desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); 
    sp32cl256pca0(i, :)=ST_VLMPF_abs(pca0_desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); 
    
    v256pca256(i, :)=VLAD_1(pca256_desc, cell_Clusters{3}.vocabulary);
    v256pca0(i, :)=VLAD_1(pca0_desc, cell_Clusters{4}.vocabulary);


         if i == 1
             descParamUsed
         end
         
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

nEncoding=6;
allDist=cell(1, nEncoding);



temp=NormalizeRowsUnit(sp32cl256pca64); allDist{1}=temp * temp';
temp=NormalizeRowsUnit(sp32cl256pca128); allDist{2}=temp * temp';
temp=NormalizeRowsUnit(sp32cl256pca256); allDist{3}=temp * temp';
temp=NormalizeRowsUnit(sp32cl256pca0); allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v256pca256, alpha)); allDist{5}=temp * temp';
temp=NormalizeRowsUnit(PowerNormalization(v256pca0, alpha)); allDist{6}=temp * temp';

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
end

delete(gcp('nocreate')) %///
%%%%

finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('%.3f\n', finalAcc(j));

    
end

fileID=fopen('dimPCArezults.txt', 'a');
fprintf(fileID, '%s\n', DescParam2Name(descParam));
fprintf('sp32cl256pca64: %.3f   sp32cl256pca128: %.3f    sp32cl256pca256: %.3f    sp32cl256pca0: %.3f    v256pca256: %.3f    v256pca0: %.3f   \n\n', ...
        finalAcc(1), finalAcc(2), finalAcc(3), finalAcc(4), finalAcc(5), finalAcc(6));
fclose(fileID);



 bazeSavePath='/home/ionut/asustor_ionut/Data/results/cvpr2017/hmdb51/dimPCA/';
 
 
saveName = [bazeSavePath 'clfsOut/'  DescParam2Name(descParam) '.mat']
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy', 'mean_all_accuracy');

saveName2 = [bazeSavePath 'videoRep/'  DescParam2Name(descParam) '.mat']
save(saveName2, '-v7.3', 'descParam', 'sp32cl256pca64', 'sp32cl256pca128', 'sp32cl256pca256', 'sp32cl256pca0', 'v256pca256', 'v256pca0');

 