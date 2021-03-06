function  [all_accuracy, all_clfsOut]  = FisherVectorFramework_hmdb51_UCF101__noSpatialPyramid(typeFeature,mType, normStrategy,datasetName,cl, nPar, alpha, savePath, fsr,  pathFeatures, d, iDTfeature, layer, Net)

addpath('./../');%!!!!!!!

global DATAopts;


% Parameter settings for descriptor extraction
clear descParam
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
                                                descParam.numClusters, descParam.pcaDim, 1000000, 1); 
                                                                                       

[tDesc] = MediaName2Descriptor(allPathFeatures{1}, descParam, pcaMap);
tDesc=tDesc';
t=mexFisherAssign(tDesc, gmmModelName)';
    
    
fisherVector=zeros(length(allPathFeatures), length(t), 'like', t);

parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(allPathFeatures));
parfor i=1:length(allPathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(allPathFeatures{i}, descParam, pcaMap);
    desc =  desc';
    
        % Feature vector assignment with spatial pyramid
    %featSpIdx = SpatialPyramidSeparationIdx(info, sRow, sCol)';
    
    
    fisherVector(i, :)=mexFisherAssign(desc, gmmModelName)';

    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=1;
allDist=cell(1, nEncoding);

n_fisherVector=NormalizeRowsUnit(PowerNormalization(fisherVector, alpha));
allDist{1}=n_fisherVector * n_fisherVector';
clear n_fisherVector

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
    
    try    
    
        
        
        fileName=sprintf('%s%s%s_fisherVector.txt',savePath(1:strfind(savePath, '%')-1), descParam.Dataset,descParam.MediaType)

        fileID=fopen(fileName, 'a');

        fprintf(fileID, '%s PNL2(alpha=%.2f)--> %.3f \n', ...
                DescParam2Name(descParam), alpha, finalAcc(j) );

        fclose(fileID);
    
    catch err
    
        fileName=sprintf('%s%s_fisherVector.txt', descParam.Dataset,descParam.MediaType)

        fileID=fopen(fileName, 'a');

        fprintf(fileID, '%s PNL2(alpha=%.2f)--> %.3f \n', ...
                DescParam2Name(descParam), alpha, finalAcc(j) );

        fclose(fileID);

        warning('error writing %s. Instead the file%s was saved in: ',err, fileName);
        
    end
end



saveName=sprintf(savePath, 'clfsOut');
saveName=sprintf('%s%sfisherVector.mat', saveName, DescParam2Name(descParam))
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy', 'mean_all_accuracy');


saveName2=sprintf(savePath, 'videoRep');
saveName2=sprintf('%s%sfisherVector.mat', saveName2, DescParam2Name(descParam))
save(saveName2, '-v7.3', 'descParam', 'fisherVector');
 
end
