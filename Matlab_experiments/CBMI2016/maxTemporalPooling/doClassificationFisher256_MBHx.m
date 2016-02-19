% % function doClassificationFisher
% 
% global DATAopts;
% DATAopts = UCFInit;
% 
% % Parameter settings for descriptor extraction
% clear descParam
% descParam.Func = @FEVidMBHxDense;
% descParam.BlockSize = [8 8 6];
% descParam.NumBlocks = [3 3 2];
% descParam.MediaType = 'Vid';
% descParam.NumOr = 8;
% descParam.Normalisation='ROOTSIFT';
% %descParam.FrameSampleRate = 1;
% %descParam.ColourSpace = colourSpace
% descParam.Pooling='maxTempPool';
% 
% sRow = [1 3];
% sCol = [1 1];
% 
% 
% descParam.pcaDim = 72;
% descParam.numClusters = 256;
% 
% descParam
% 
% 
% 
% vocabularyIms = GetVideosPlusLabels('smallEnd');
% 
% [gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyIms, descParam, ...
%                                                 descParam.numClusters, descParam.pcaDim);
% 
% % Now create set
% [vids, labs, groups] = GetVideosPlusLabels('Full');
% 
% 
% 
%     [tDesc] = MediaName2Descriptor(vids{1}, descParam, pcaMap);
%     tDesc = tDesc'; 
%     tFisher=mexFisherAssign(tDesc, gmmModelName)';
%     
% fisher1=zeros(length(vids), length(tFisher), 'like', tFisher);
% fisher2=zeros(length(vids), length(tFisher), 'like', tFisher);
% fisher3=zeros(length(vids), length(tFisher), 'like', tFisher);
% fisher4=zeros(length(vids), length(tFisher), 'like', tFisher);
% 
% 
% % Now object visual word frequency histograms
% fprintf('Descriptor extraction  for %d vids: ', length(vids));
% for i=1:length(vids)
%     fprintf('%d ', i)
%     % Extract descriptors
%     
%     [desc, info, descParamUsed] = MediaName2Descriptor(vids{i}, descParam, pcaMap);
%     desc = desc';
%     
%         % Feature vector assignment with spatial pyramid
%     featSpIdx = SpatialPyramidSeparationIdx(info, sRow, sCol)';
%     fisherVT = cell(1,size(featSpIdx,1));
%     
%     fisher1(i, :)=mexFisherAssign(desc(:,featSpIdx(1,:)), gmmModelName)';
%     fisher2(i, :)=mexFisherAssign(desc(:,featSpIdx(2,:)), gmmModelName)';
%     fisher3(i, :)=mexFisherAssign(desc(:,featSpIdx(3,:)), gmmModelName)';
%     fisher4(i, :)=mexFisherAssign(desc(:,featSpIdx(4,:)), gmmModelName)';
%     
%         
%          if i == 1
%              descParamUsed
%          end
%          
% end
% fprintf('\nDone!\n');
% 
% 
% 
% 
% %% Do classification
% 
% nEncoding=2;
% allDist=cell(1, nEncoding);
% 
% fisherAll=cat(2, fisher1,fisher2, fisher3, fisher4);


n_fisherAll=NormalizeRowsUnit(PowerNormalization(fisherAll, 0.5));
allDist{1}=n_fisherAll * n_fisherAll';
clear n_fisherAll

n_fisher1=NormalizeRowsUnit(PowerNormalization(fisher1, 0.5));
allDist{2}=n_fisher1 * n_fisher1';
clear n_fisher1

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


for k=1:nEncoding

% 
% Leave-one-group-out cross-validation
for i=1:max(groups)
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

mean(perGroupAccuracy)
end


% v1_meanAcc=mean(mean(cat(2, all_accuracy{1}{:})));
% v2_meanAcc=mean(mean(cat(2, all_accuracy{2}{:})));

saveName = ['/data/ionut/rezults/CBMI2016/' 'clfsOut/' DescParam2Name(descParam) '_sRow3_Fisher_.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = ['/data/ionut/rezults/CBMI2016/' 'videoRep/' DescParam2Name(descParam) '_sRow3_Fisher_.mat'];
 save(saveName2, '-v7.3', 'fisherAll');
