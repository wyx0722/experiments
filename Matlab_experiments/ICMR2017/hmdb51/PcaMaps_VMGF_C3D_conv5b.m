addpath('./../');%!!!!!!!
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


d=[32 64 128 256 512];

nPar=5;



descParam.Dataset=datasetName;
descParam.MediaType=mType;
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;


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


cell_pcaMap=CreatePcaMaps(vocabularyPathFeatures, descParam);

[desc, info, descParam] = descParam.Func(allPathFeatures{1}, descParam);


desc32 = desc * cell_pcaMap{1}.data.rot;
desc64 = desc * cell_pcaMap{2}.data.rot;
desc128 = desc * cell_pcaMap{3}.data.rot;
desc256 = desc * cell_pcaMap{4}.data.rot;
desc512 = desc * cell_pcaMap{5}.data.rot;



t=VMGF_abs(desc32); vmgf32=zeros(length(allPathFeatures), length(t), 'like', t);
t=VMGF_abs_2(desc32, desc); vmgf32_2=zeros(length(allPathFeatures), length(t), 'like', t);

t=VMGF_abs(desc64); vmgf64=zeros(length(allPathFeatures), length(t), 'like', t);
t=VMGF_abs_2(desc64, desc); vmgf64_2=zeros(length(allPathFeatures), length(t), 'like', t);

t=VMGF_abs(desc128); vmgf128=zeros(length(allPathFeatures), length(t), 'like', t);
t=VMGF_abs_2(desc128, desc); vmgf128_2=zeros(length(allPathFeatures), length(t), 'like', t);

t=VMGF_abs(desc256); vmgf256=zeros(length(allPathFeatures), length(t), 'like', t);
t=VMGF_abs_2(desc256, desc); vmgf256_2=zeros(length(allPathFeatures), length(t), 'like', t);

t=VMGF_abs(desc512); vmgf512=zeros(length(allPathFeatures), length(t), 'like', t);
t=VMGF_abs_2(desc512, desc); vmgf512_2=zeros(length(allPathFeatures), length(t), 'like', t);

clear desc32  desc64 desc128 desc256 desc512

nDesc=zeros(1, length(allPathFeatures));

parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(allPathFeatures));
parfor i=1:length(allPathFeatures)%parfor i=1:length(allPathFeatures)
%     if mod(i, 100)==0
%         fprintf('%d ', i)%fprintf('%d \n', i)
%     end
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = descParam.Func(allPathFeatures{i}, descParam);
    nDesc(i)=size(desc,1);
    
    desc32 = desc * cell_pcaMap{1}.data.rot;
    desc64 = desc * cell_pcaMap{2}.data.rot;
    desc128 = desc * cell_pcaMap{3}.data.rot;
    desc256 = desc * cell_pcaMap{4}.data.rot;
    desc512 = desc * cell_pcaMap{5}.data.rot;



    vmgf32(i, :)=VMGF_abs(desc32); 
    vmgf32_2(i, :)=VMGF_abs_2(desc32, desc); 

    vmgf64(i, :)=VMGF_abs(desc64); 
    vmgf64_2(i, :)=VMGF_abs_2(desc64, desc); 

    vmgf128(i, :)=VMGF_abs(desc128);
    vmgf128_2(i, :)=VMGF_abs_2(desc128, desc);

    vmgf256(i, :)=VMGF_abs(desc256); 
    vmgf256_2(i, :)=VMGF_abs_2(desc256, desc);

    vmgf512(i, :)=VMGF_abs(desc512); 
    vmgf512_2(i, :)=VMGF_abs_2(desc512, desc); 


    
         if i == 1
             descParamUsed
         end
         
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

nEncoding=10;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(vmgf32); allDist{1}=temp * temp';
temp=NormalizeRowsUnit(vmgf32_2); allDist{2}=temp * temp';

temp=NormalizeRowsUnit(vmgf64); allDist{3}=temp * temp';
temp=NormalizeRowsUnit(vmgf64_2); allDist{4}=temp * temp';

temp=NormalizeRowsUnit(vmgf128); allDist{5}=temp * temp';
temp=NormalizeRowsUnit(vmgf128_2); allDist{6}=temp * temp';

temp=NormalizeRowsUnit(vmgf256); allDist{7}=temp * temp';
temp=NormalizeRowsUnit(vmgf256_2); allDist{8}=temp * temp';

temp=NormalizeRowsUnit(vmgf512); allDist{9}=temp * temp';
temp=NormalizeRowsUnit(vmgf512_2); allDist{10}=temp * temp';
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

