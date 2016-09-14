
basePath='/home/ionut/asustor_ionut/'


%!!!!!!!change both!!!!!!!!!!!!!!!!!!!!!
savePath=[basePath 'Data/results/mtap2017/ucf101/%s/iDT/'];
pathFeatures=[basePath 'Data/iDT_Features_UCF101/Videos/']



addpath('./../');%!!!!!!!

global DATAopts;


% Parameter settings for descriptor extraction
clear descParam
descParam.Dataset='UCF101';
descParam.MediaType='IDT';
descParam.Func = @FEVid_IDT;
descParam.Normalisation='ROOTSIFT';
descParam.numClusters = 256;
alpha=0.1%!!!!!!!change

 iDTfeature='HOG';
 
 
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
        %sRow = [1 3];
        %sCol = [1 1];
        
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
        allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4) '/' descParam.Layer '.txt'];
    elseif ~isempty(strfind(descParam.MediaType, 'IDT'))
        if ~isempty(strfind(descParam.Dataset, 'UCF101'))
            allPathFeatures{i}=[bazePathFeatures allVids{i}];
        else
            allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4)];
        end
    elseif ~isempty(strfind(descParam.MediaType, 'Vid'))
        
        if ~isempty(strfind(descParam.Dataset, 'HMDB51'))
            allPathFeatures{i}=[DATAopts.videoPath, allVids{i}];
        else
            allPathFeatures{i}=sprintf(DATAopts.videoPath, allVids{i});
        end
    end
end

vocabularyPathFeatures=allPathFeatures(1:4:end);%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


[gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyPathFeatures, descParam, ...
                                                descParam.numClusters, descParam.pcaDim); 
                   
