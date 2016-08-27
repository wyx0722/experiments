function  [all_accuracy, all_clfsOut]  = FisherFramework(typeFeature, normStrategy, d, cl, fsr, nPar, alpha)

% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;


addpath('./../');%!!!!!!!


% Parameter settings for descriptor extraction
clear descParam
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;
descParam.pcaDim = d;
descParam.numClusters = cl;

descParam.NumBlocks = [3 3 2];

if nargin>4
descParam.FrameSampleRate = fsr;
descParam.BlockSize = [8 8 6/fsr];

else
    descParam.BlockSize = [8 8 6];
end

descParam.MediaType = 'Vid';
descParam.NumOr = 8;

%descParam.FrameSampleRate = 1;
%descParam.ColourSpace = colourSpace

sRow = [1 3];
sCol = [1 1];





descParam



vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=sprintf(DATAopts.videoPath, vocabularyIms{i});
end



[gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim);

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
fullPathVids=cell(size(vids));

for i=1:length(fullPathVids)
    fullPathVids{i}=sprintf(DATAopts.videoPath, vids{i});
end



    [tDesc] = MediaName2Descriptor(fullPathVids{1}, descParam, pcaMap);
    tDesc = tDesc'; 
    tFisher=mexFisherAssign(tDesc, gmmModelName)';
    
fisher1=zeros(length(vids), length(tFisher), 'like', tFisher);
fisher2=zeros(length(vids), length(tFisher), 'like', tFisher);
fisher3=zeros(length(vids), length(tFisher), 'like', tFisher);
fisher4=zeros(length(vids), length(tFisher), 'like', tFisher);

parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(fullPathVids));
parfor i=1:length(fullPathVids)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(fullPathVids{i}, descParam, pcaMap);
    desc = desc';
    
        % Feature vector assignment with spatial pyramid
    featSpIdx = SpatialPyramidSeparationIdx(info, sRow, sCol)';
    
    
    fisher1(i, :)=mexFisherAssign(desc(:,featSpIdx(1,:)), gmmModelName)';
    fisher2(i, :)=mexFisherAssign(desc(:,featSpIdx(2,:)), gmmModelName)';
    fisher3(i, :)=mexFisherAssign(desc(:,featSpIdx(3,:)), gmmModelName)';
    fisher4(i, :)=mexFisherAssign(desc(:,featSpIdx(4,:)), gmmModelName)';
    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=2;
allDist=cell(1, nEncoding);

fisherAll=cat(2, fisher1,fisher2, fisher3, fisher4);


temp=NormalizeRowsUnit(PowerNormalization(fisherAll, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(fisherAll, descParam.pcaDim), alpha));
allDist{2}=temp * temp';

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


for k=1:nEncoding

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

k
perGroupAccuracy = mean(cat(2, accuracy{:}))'

end

delete(gcp('nocreate'))




hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if ~isempty(findstr(hostname, 'cocoa'))
    rezPath='/home/ionut/asustor_ionut/Data/results/mtap2017/'
else if ~isempty(findstr(hostname, 'Halley'))
      rezPath='/home/ionut/asustor_ionut_2/Data/results/mtap2017/'
    end
end


try    
    
    fileName=[rezPath  'resultsDenseFisherVector.txt'];
    
    fileID=fopen(fileName, 'a');
    
    fprintf(fileID, '%s PNL2 norm before classification (alpha=%.2f) \n  FisherVector: %.3f   FisherVector_intraL2: %.3f \n\n', ...
            DescParam2Name(descParam), alpha, mean(mean(cat(2, all_accuracy{1}{:}))), mean(mean(cat(2, all_accuracy{2}{:}))) );
    
    fclose(fileID);
    
catch err
    
    fileName='resultsDenseFisherVector.txt';
    fileID=fopen(fileName, 'a');
    
   fprintf(fileID, '%s PNL2 norm before classification (alpha=%.2f) \n  FisherVector: %.3f   FisherVector_intraL2: %.3f \n\n', ...
            DescParam2Name(descParam), alpha, mean(mean(cat(2, all_accuracy{1}{:}))), mean(mean(cat(2, all_accuracy{2}{:}))) );
    
    fclose(fileID);
    
    warning('error writing %s. Instead the file%s was saved in: ',err, fileName);
        
end

    
saveName = [rezPath 'clfsOut/'  DescParam2Name(descParam) '.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = [rezPath 'videoRep/' DescParam2Name(descParam) '.mat'];
 save(saveName2, '-v7.3', 'descParam', 'fisherAll');
end
