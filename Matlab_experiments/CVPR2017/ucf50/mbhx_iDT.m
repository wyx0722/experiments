% addpath('./../');%!!!!!!!
% addpath('./../../');%!!!!!!!


global DATAopts;
DATAopts = UCFInit;

datasetName='UCF50';
mType='IDT';
typeFeature=@FEVid_IDT;
normStrategy='ROOTSIFT';
iDTfeature='MBHx'
cl=256;
%~~~~~~~~~~~~~~

allPathFeatures='/home/ionut/asustor_ionut/Data/Features/UCF50/IDT/Videos/'%%%%%channge~~~~~~~~~~~~
%~~~~~~~~~~~~~~


alpha=0.1;
nPar=5;



descParam.Dataset=datasetName;
descParam.MediaType=mType;
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;
descParam.numClusters = cl;



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
        bazePathFeatures=allPathFeatures
        
    case 'DeepF'
        descParam.Layer=layer;
        descParam.net=Net;
 
        sizeDesc=512;
        if exist('d', 'var')
            descParam.pcaDim=d;
        else
            descParam.pcaDim=sizeDesc/2;
        end
        bazePathFeatures=allPathFeatures
end


descParam







vocabularyIms = GetVideosPlusLabels('smallEnd');
vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    if ~isempty(strfind(descParam.MediaType, 'DeepF'))
        if ~isempty(strfind(descParam.net, 'C3D'))
            file_extension='.mat';
        else
            file_extension='.txt';
        end
        vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '/' descParam.Layer file_extension];
    elseif strfind(descParam.MediaType, 'IDT')>0 
        vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i))];
    end
end


%vocabularyPathFeatures=allPathFeatures(1:4:end);%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


[gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim);

% Now create set
[allVids, labs, groups] = GetVideosPlusLabels('Full');
allPathFeatures=cell(size(allVids));

for i=1:length(allPathFeatures)
    if ~isempty(strfind(descParam.MediaType, 'DeepF')) 
         if ~isempty(strfind(descParam.net, 'C3D'))
            file_extension='.mat';
        else
            file_extension='.txt';
        end
        allPathFeatures{i}=[bazePathFeatures char(allVids(i)) '/' descParam.Layer file_extension];
    elseif strfind(descParam.MediaType, 'IDT')>0  
        allPathFeatures{i}=[bazePathFeatures char(allVids(i))];
    end
end


    [tDesc] = MediaName2Descriptor(allPathFeatures{1}, descParam, pcaMap);
    tDesc = tDesc'; 
    tFisher=mexFisherAssign(tDesc, gmmModelName)';
    
fisherAll=zeros(length(allPathFeatures), length(tFisher), 'like', tFisher);


nDesc=zeros(1, length(allPathFeatures));

%parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(allPathFeatures));
for i=1:length(allPathFeatures)%parfor i=1:length(allPathFeatures)
    if mod(i, 100)==0
        fprintf('%d ', i)%fprintf('%d \n', i)
    end
    
    %fprintf('%d \n', i)
    % Extract descriptors
    

    [desc, info, descParamUsed] = MediaName2Descriptor(allPathFeatures{i}, descParam, pcaMap);
    desc = desc';
    nDesc(i)=size(desc,1);
    
    fisherAll(i, :)=mexFisherAssign(desc, gmmModelName)';



         if i == 1
             descParamUsed
         end
         
end
%delete(gcp('nocreate'))
fprintf('\nDone!\n');

nEncoding=8;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(PowerNormalization(fisherAll, alpha));
allDist{1}=temp * temp';

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


for k=1:nEncoding
k
% 
% Leave-one-group-out cross-validation
parfor i=1:max(groups)
    testI = groups == i;
    trainI = ~testI;
    trainDist = allDist{k}(trainI, trainI);
    testDist = allDist{k}(testI, trainI);
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

all_clfsOut{k}=clfsOut;
all_accuracy{k}=accuracy;

fprintf('Accuracy for encoding %d: %.3f\n',k, mean(mean(cat(2, accuracy{:}))'));

end

delete(gcp('nocreate'))

clear allDist



finalAcc=zeros(1,nEncoding);
for j=1:nEncoding

    finalAcc(j)=mean(mean(cat(2, all_accuracy{j}{:}), 2));
    fprintf('%.3f\n', finalAcc(j));

    
end

 bazeSavePath='/home/ionut/asustor_ionut/Data/results/cvpr2017/ucf50/';
 
 
    
saveName = [bazeSavePath 'clfsOut/'  DescParam2Name(descParam) '_fisherAll__all_accuracy.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = [bazeSavePath 'videoRep/' DescParam2Name(descParam) '__fisherAll.mat'];
 save(saveName2, '-v7.3', 'descParam', 'fisherAll');
