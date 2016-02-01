function  [all_accuracy, all_clfsOut]  = VLAD256_doClassification_IDT_SP(typeFeature, normStrategy, alpha)


% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam

%descParam.BlockSize = [8 8 6];
%descParam.NumBlocks = [3 3 2];
%descParam.NumOr = 8;
%descParam.FrameSampleRate = 1;
%descParam.ColourSpace = colourSpace

descParam.Func = @FEVid_IDT; %descParam.Func = @FEVidHOF_IDT;
descParam.MediaType = 'IDT';
descParam.IDTfeature=typeFeature; %descParam.IDTfeature='HOF';
descParam.Normalisation=normStrategy; %descParam.Normalisation='L1PN'; % L2 or 'ROOTSIFT'

descParam.sRow=3;
descParam.sCol=1;


if nargin>2
    descParam.alpha=alpha; %descParam.alpha=0.3;
else
    descParam.alpha=1;
end

%sRow = [1 3];
%sCol = [1 1];

if strcmp(descParam.IDTfeature,'HOF') || strcmp(descParam.IDTfeature,'HOF_iTraj')
    sizeDesc=108;
    
elseif  strcmp(descParam.IDTfeature,'HOG') || strcmp(descParam.IDTfeature,'MBHx') || strcmp(descParam.IDTfeature,'MBHy') ...
        || strcmp(descParam.IDTfeature,'HOG_iTraj') || strcmp(descParam.IDTfeature,'MBHx_iTraj') || strcmp(descParam.IDTfeature,'MBHy_iTraj')
    sizeDesc=96;   
end

% pcaDim & vocabulary size
descParam.pcaDim = sizeDesc/2;
descParam.numClusters = 256;

descParam

%%%%%%%%%%
bazePathFeatures='/home/ionut/Features/Features/UCF50/IDT/Videos/'; %change

vocabularyIms = GetVideosPlusLabels('smallEnd');
vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i))];
end
    
%[gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
%                                                numClusters, pcaDim);
[vocabulary, pcaMap, st_d, skew, nElem] = CreateVocabularyKmeansPca_m(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim);
                                     


% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');

pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i))];
end


useSP=1;  %change to zero if Spatial Pyramid is not used

if (useSP==1)
    nDivImg=descParam.sRow * descParam.sCol;
    if nDivImg>1
        nDivImg=nDivImg+1;
    end

    dimVlad = size(vocabulary, 1) * size(vocabulary, 2) * nDivImg;

else
    dimVlad=size(vocabulary, 1) * size(vocabulary, 2);
end

nEncoding=3;

vladVectors1= zeros(length(vids), dimVlad);  
vladVectors2= zeros(length(vids), 2*dimVlad);
vladVectors3= zeros(length(vids), 3*dimVlad);

len_v1=size(vladVectors1, 2);
len_v2=size(vladVectors2, 2);
len_v3=size(vladVectors3, 2);



% Now object visual word frequency histograms
fprintf('IDT VLAD extraction  for %d vids: ', length(pathFeatures));
for i=1:length(pathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
    
        
        [idx] = SpatialPyramid_Rows(info.infoTraj, descParam.sRow);
        
        for j=1:size(idx, 2)
            
            pozB1 = j*(len_v1/nDivImg) - (len_v1/nDivImg) +1;
            pozE1 = j*(len_v1/nDivImg); 
            vladVectors1(i, pozB1:pozE1) = VLAD_1(desc(idx(:,j), :), vocabulary);
            
            pozB2 = j*(len_v2/nDivImg) - (len_v2/nDivImg) +1;
            pozE2 = j*(len_v2/nDivImg);
            vladVectors2(i, pozB2:pozE2) = improvedVLAD(desc(idx(:,j), :), vocabulary, st_d, skew, nElem);
            
            pozB3 = j*(len_v3/nDivImg) - (len_v3/nDivImg) +1;
            pozE3 = j*(len_v3/nDivImg);
            vladVectors3(i, pozB3:pozE3) = BoostingVLAD_paper(desc(idx(:,j), :), vocabulary, st_d, skew, nElem);
        end
       
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');

%% Do classification

allDist=cell(1, nEncoding);

intraN_vladVectors1=intranormalizationFeatures(vladVectors1, size(vocabulary, 2));
intraN_vladVectors2=intranormalizationFeatures(vladVectors2, size(vocabulary, 2));
intraN_vladVectors3=intranormalizationFeatures(vladVectors3, size(vocabulary, 2));


n_vladVectors1=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors1, 0.5));
allDist{1}=n_vladVectors1 * n_vladVectors1';
clear n_vladVectors1

n_vladVectors2=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors2, 0.5));
allDist{2}=n_vladVectors2 * n_vladVectors2';
clear n_vladVectors2

n_vladVectors3=NormalizeRowsUnit(PowerNormalization(intraN_vladVectors3, 0.5));
allDist{3}=n_vladVectors3 * n_vladVectors3';
clear n_vladVectors3

clear intraN_vladVectors1
clear intraN_vladVectors2
clear intraN_vladVectors3


all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;

parpool(5);

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



%%%%%%%%%%%%save results
v1_meanAcc=mean(mean(cat(2, all_accuracy{1}{:})));
v2_meanAcc=mean(mean(cat(2, all_accuracy{2}{:})));
v3_meanAcc=mean(mean(cat(2, all_accuracy{3}{:})));

try    
    
    fileName=['/home/ionut/Data/results_desc_IDT/rez_SpatialPyramid/results_UCF50_SpatialPyramid3Rows__' 'Desc' descParam.MediaType '_' descParam.IDTfeature '_norm' descParam.Normalisation '.txt'];
    
    fileID=fopen(fileName, 'a');
    
    fprintf(fileID, '%s %s norm:%s  alpha: %.2f --> v1: %.3f  v2: %.3f  v3: %.3f \r\n', ...
            descParam.MediaType,descParam.IDTfeature, descParam.Normalisation,descParam.alpha,v1_meanAcc,v2_meanAcc, v3_meanAcc);
    
    fclose(fileID);
    
catch err
    
    warning('error writing %s.: ',err);
        
end



saveName = ['/home/ionut/Data/results_desc_IDT/' 'clfsOut/' 'SpatialPyramid/' DescParam2Name(descParam) '_VLAD_.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = ['/home/ionut/Data/results_desc_IDT/' 'videoRep/' 'SpatialPyramid/' DescParam2Name(descParam) '_VLAD_.mat'];
 save(saveName2, '-v7.3', 'vladVectors1', 'vladVectors2', 'vladVectors3');

end
