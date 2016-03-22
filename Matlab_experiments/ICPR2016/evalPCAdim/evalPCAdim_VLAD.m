function [all_accuracy, all_clfsOut]=evalPCAdim_VLAD(func, mediaType, layer, network, norm, nCl, pcaD, bPathFeatures, dAssign, nPar)

global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = func;
descParam.MediaType = mediaType;
descParam.Layer=layer;
descParam.net=network;
descParam.Normalisation=norm;

descParam.numClusters = nCl;

descParam.pcaDim = pcaD;

descParam.DoubleAssign=dAssign;

descParam

%%%%%%%%%%
%bazePathFeatures='/home/ionut/halley_ionut/Data/VGG_19_v-features_rawFrames_UCF50/Videos/'; %change
bazePathFeatures=bPathFeatures;

vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '/' descParam.Layer '.txt'];
end


                                            
[vocabulary, pcaMap] = CreateVocabularyKmeansPca(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim); 
                                            
%vocabulary = NormalizeRowsUnit(vocabulary); %make unit length

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i)) '/' descParam.Layer '.txt'];
end



    [tDesc] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
    %tDesc=NormalizeRowsUnit(tDesc);
    tVLAD=doubleAssign_VLAD_1(tDesc, vocabulary, 1);
    
vlad=zeros(length(vids), length(tVLAD), 'like', tVLAD);


parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
parfor i=1:length(pathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
   % desc = NormalizeRowsUnit(desc);
   
    
    
    vlad(i,: )=doubleAssign_VLAD_1(desc, vocabulary, descParam.DoubleAssign);

    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=1;
allDist=cell(1, nEncoding);

n_vlad=NormalizeRowsUnit(PowerNormalization(vlad, 0.5));
allDist{1}=n_vlad * n_vlad';
clear n_vlad



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


fileName=sprintf('/home/ionut/experiments/Matlab_experiments/ICPR2016/results/evalPCAdim/results_UCF50_evalPCAdim_Features%s_Layer%s_Network%s_Clusters%d_DoubleAssign%f_norm%s_PNL05__VLAD.txt', descParam.MediaType,descParam.Layer, descParam.net,descParam.numClusters,descParam.DoubleAssign, descParam.Normalisation); 
fileID=fopen(fileName, 'a');
acc=mean(mean(cat(2, all_accuracy{1}{:})))
fprintf(fileID, 'PCAdim  %d --> acc= %.4f \r\n', descParam.pcaDim, acc);
fclose(fileID);



saveName = ['/home/ionut/Data/results/ICPR2016_rezults/' 'clfsOut/' 'dimPCA/'  DescParam2Name(descParam) '_VLAD_.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = ['/home/ionut/Data/results/ICPR2016_rezults/' 'videoRep/' 'dimPCA/' DescParam2Name(descParam) '_VLAD_.mat'];
 save(saveName2, '-v7.3', 'vlad');
end