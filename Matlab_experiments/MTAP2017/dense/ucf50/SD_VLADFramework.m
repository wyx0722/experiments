function  [all_accuracy, all_clfsOut]  = SD_VLADFramework(typeFeature, normStrategy, d, cl, fsr, nPar, alpha)

addpath('./../');%!!!!!!!

global DATAopts;
DATAopts = UCFInit;

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



                                            
[vocab, pcaMap] = CreateVocabularyKmeansPca_mSingle(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim); 
nV_vocab=vocab;
nV_vocab.vocabulary=NormalizeRowsUnit(nV_vocab.vocabulary);


% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
fullPathVids=cell(size(vids));

for i=1:length(fullPathVids)
    fullPathVids{i}=sprintf(DATAopts.videoPath, vids{i});
end



[tDesc] = MediaName2Descriptor(fullPathVids{1}, descParam, pcaMap);
tDesc=NormalizeRowsUnit(tDesc);
t=SD_VLAD_fast(tDesc, nV_vocab.vocabulary, nV_vocab.st_d);
    
    
sd_vlad1=zeros(length(vids), length(t), 'like', t);
sd_vlad2=zeros(length(vids), length(t), 'like', t);
sd_vlad3=zeros(length(vids), length(t), 'like', t);
sd_vlad4=zeros(length(vids), length(t), 'like', t);

parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(fullPathVids));
parfor i=1:length(fullPathVids)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(fullPathVids{i}, descParam, pcaMap);
    desc = NormalizeRowsUnit(desc);
    
        % Feature vector assignment with spatial pyramid
    featSpIdx = SpatialPyramidSeparationIdx(info, sRow, sCol)';
    
    
    sd_vlad1(i, :)=SD_VLAD_fast(desc(featSpIdx(1,:), :), nV_vocab.vocabulary, nV_vocab.st_d);
    sd_vlad2(i, :)=SD_VLAD_fast(desc(featSpIdx(2,:), :), nV_vocab.vocabulary, nV_vocab.st_d);
    sd_vlad3(i, :)=SD_VLAD_fast(desc(featSpIdx(3,:), :), nV_vocab.vocabulary, nV_vocab.st_d);
    sd_vlad4(i, :)=SD_VLAD_fast(desc(featSpIdx(4,:), :), nV_vocab.vocabulary, nV_vocab.st_d);
    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=1;
allDist=cell(1, nEncoding);

SD_VLADAll=cat(2, sd_vlad1,sd_vlad2, sd_vlad3, sd_vlad4);

n_VLADAll=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(SD_VLADAll, descParam.pcaDim), alpha));
allDist{1}=n_VLADAll * n_VLADAll';
clear n_VLADAll

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
    
    fileName=[rezPath  'resultsDenseFrameSR.txt'];
    
    fileID=fopen(fileName, 'a');
    
    fprintf(fileID, '%s PNL2 norm before classification (alpha=%.2f) \n  SD-VLAD: %.3f    \n\n', ...
            DescParam2Name(descParam), alpha, mean(mean(cat(2, all_accuracy{1}{:}))) );
    
    fclose(fileID);
    
catch err
    
    fileName='resultsDenseFrameSR.txt';
    fileID=fopen(fileName, 'a');
    
      fprintf(fileID, '%s PNL2 norm before classification (alpha=%.2f) \n  SD-VLAD: %.3f    \n\n', ...
            DescParam2Name(descParam), alpha, mean(mean(cat(2, all_accuracy{1}{:}))) );
    
    fclose(fileID);
    
    warning('error writing %s. Instead the file%s was saved in: ',err, fileName);
        
end

    
saveName = [rezPath 'clfsOut/'  DescParam2Name(descParam) '.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = [rezPath 'videoRep/' DescParam2Name(descParam) '.mat'];
 save(saveName2, '-v7.3', 'descParam', 'SD_VLADAll');
end
