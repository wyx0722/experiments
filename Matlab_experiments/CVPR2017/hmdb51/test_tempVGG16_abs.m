addpath('./../');%!!!!!!!

global DATAopts;

datasetName='HMDB51';
mType='DeepF';
typeFeature=@FEVid_deepFeatures;
normStrategy='None';
layer='pool5'
Net='TempSplit1VGG16';

alpha=0.5;
nPar=5;
pathFeatures='/home/ionut/asustor_ionut/Data/hmdb51_action_temporal_vgg_16_split1_features_opticalFlow_tvL1/Videos/'%%%%%channge~~~~~~~~~~~~


descParam.Dataset=datasetName;
descParam.MediaType=mType;
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;

descParam.Clusters=[256 319 512];
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

[cell_Clusters, cell_spClusters, pcaMap] = CreateVocabularyKmeansPca_sptCl(vocabularyPathFeatures, descParam);
 
[tDesc, info] = MediaName2Descriptor(allPathFeatures{1}, descParam, pcaMap);                                           

t=VLAD_1(tDesc, cell_Clusters{1}.vocabulary);
v256=zeros(length(allPathFeatures), length(t), 'like', t); 

t=VLAD_1(tDesc, cell_Clusters{2}.vocabulary);
v319=zeros(length(allPathFeatures), length(t), 'like', t); 

t=VLAD_1_mean(tDesc, cell_Clusters{3}.vocabulary);
v512=zeros(length(allPathFeatures), length(t), 'like', t);

t=max_pooling(tDesc, cell_Clusters{1}.vocabulary);
m256_abs=zeros(length(allPathFeatures), length(t), 'like', t);

t=max_pooling(tDesc, cell_Clusters{2}.vocabulary);
m319_abs=zeros(length(allPathFeatures), length(t), 'like', t); 

t=max_pooling(tDesc, cell_Clusters{3}.vocabulary);
m512_abs=zeros(length(allPathFeatures), length(t), 'like', t);


t=VLAD_1_mean_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
spV32_m=zeros(length(allPathFeatures), length(t), 'like', t);

t=VLAD_1_spClustering_memb(tDesc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
spV32=zeros(length(allPathFeatures), length(t), 'like', t);

t=ST_VLMPF(tDesc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
st_vlmpf32_abs=zeros(length(allPathFeatures), length(t), 'like', t);

nDesc=zeros(1, length(allPathFeatures));

parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(allPathFeatures));
parfor i=1:length(allPathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(allPathFeatures{i}, descParam, pcaMap);
    nDesc(i)=size(desc,1);
    
    v256 (i, :) = VLAD_1(desc, cell_Clusters{1}.vocabulary);
    v319 (i, :) = VLAD_1(desc, cell_Clusters{2}.vocabulary);
    v512 (i, :) = VLAD_1(desc, cell_Clusters{3}.vocabulary);
    
    m256_abs (i, :) = max_pooling_abs(desc, cell_Clusters{1}.vocabulary);
    m319_abs  (i, :) = max_pooling_abs(desc, cell_Clusters{2}.vocabulary);
    m512_abs (i, :) = max_pooling_abs(desc, cell_Clusters{3}.vocabulary);
    
    spV32_m (i, :) = VLAD_1_mean_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
    spV32 (i, :) = VLAD_1_spClustering_memb(desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
    st_vlmpf32_abs (i, :) = ST_VLMPF_abs(desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
    
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');

nEncoding=9;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(PowerNormalization(v256, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v319, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v512, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(m256_abs, alpha));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(m319_abs, alpha));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(m512_abs, alpha));
allDist{6}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV32_m, alpha));
allDist{7}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV32, alpha));
allDist{8}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(st_vlmpf32_abs, alpha));
allDist{9}=temp * temp';



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




%%%
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

delete(gcp('nocreate'))
%%%%

finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end