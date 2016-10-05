addpath('./../');%!!!!!!!
addpath('./../../');%!!!!!!!


global DATAopts;

datasetName='HMDB51';
mType='DeepF';
typeFeature=@FEVid_deepFeatures;
normStrategy='None';
layer='conv5b'
Net='C3D';

d=0;

alpha=0.5;
nPar=5;
pathFeatures='/home/ionut/asustor_ionut/Data/mat_c3d_features_hmdb51/Videos/'%%%%%channge~~~~~~~~~~~~


descParam.Dataset=datasetName;
descParam.MediaType=mType;
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;

descParam.Clusters=[64 128 256 512 100 110 120 130 140 150 160 170 180 190 200 210];
descParam.spClusters=[16 20 32];


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


%all_rep=cell(1, 16);%!!!!!!!!

t=ST_VLMPF_abs(tDesc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep1=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep2=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep3=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep4=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{5}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep5=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{6}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep6=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{7}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep7=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{8}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep8=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{9}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep9=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{10}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep10=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{11}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep11=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{12}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep12=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{13}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep13=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{14}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep14=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{15}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep15=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{16}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary); rep16=zeros(length(allPathFeatures), length(t), 'like', t); 



nDesc=zeros(1, length(allPathFeatures));

%parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(allPathFeatures));
for i=1:length(allPathFeatures)%parfor i=1:length(allPathFeatures)
    if mod(i, 100)==0
        fprintf('%d ', i)%fprintf('%d \n', i)
    end
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(allPathFeatures{i}, descParam, pcaMap);
    nDesc(i)=size(desc,1);
    

    rep1(i, :) = ST_VLMPF_abs(desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep2(i, :) = ST_VLMPF_abs(desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep3(i, :) = ST_VLMPF_abs(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep4(i, :) = ST_VLMPF_abs(desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep5(i, :) = ST_VLMPF_abs(desc, cell_Clusters{5}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep6(i, :) = ST_VLMPF_abs(desc, cell_Clusters{6}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep7(i, :) = ST_VLMPF_abs(desc, cell_Clusters{7}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep8(i, :) = ST_VLMPF_abs(desc, cell_Clusters{8}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep9(i, :) = ST_VLMPF_abs(desc, cell_Clusters{9}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep10(i, :) = ST_VLMPF_abs(desc, cell_Clusters{10}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep11(i, :) = ST_VLMPF_abs(desc, cell_Clusters{11}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep12(i, :) = ST_VLMPF_abs(desc, cell_Clusters{12}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep13(i, :) = ST_VLMPF_abs(desc, cell_Clusters{13}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep14(i, :) = ST_VLMPF_abs(desc, cell_Clusters{14}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep15(i, :) = ST_VLMPF_abs(desc, cell_Clusters{15}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    rep16(i, :) = ST_VLMPF_abs(desc, cell_Clusters{16}.vocabulary, info.spInfo, cell_spClusters{3}.vocabulary);
    
         if i == 1
             descParamUsed
         end
         
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

nEncoding=32;
allDist=cell(1, nEncoding);


for i=1:16
    temp=NormalizeRowsUnit(eval(sprintf('rep%d', i)));
    allDist{1}=temp * temp';
    
    temp=NormalizeRowsUnit(eval(sprintf('rep%d(:, 1:%d)', i, size(cell_Clusters{i}.vocabulary, 1)*size(cell_Clusters{i}.vocabulary, 2) )));
    allDist{16+i}=temp * temp';
end



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
%%%
for k=1:nEncoding
    k
    for i=1:3%parfor i=1:3
        
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

%delete(gcp('nocreate')) %///
%%%%

finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end