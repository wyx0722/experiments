
global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='fc6';
descParam.net='SpSplit1VGG16';
descParam.Normalisation='ROOTSIFT';

descParam.numClusters = 256;

descParam.pcaDim = 2048;

descParam

%%%%%%%%%%
bazePathFeatures='/home/ionut/Data/action_temporal_vgg_16_split1_features_opticalFlow_tvL1_UCF50/Videos/'; %change


vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '/fc6.txt'];
end


                                            
[vocabulary, pcaMap] = CreateVocabularyKmeansPca(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim, 30000); 
                                            
%vocabulary = NormalizeRowsUnit(vocabulary); %make unit length

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i)) '/fc6.txt'];
end



    [tDesc] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
    %tDesc=NormalizeRowsUnit(tDesc);
    tVLAD=VLAD_1_mean(tDesc, vocabulary);
    
vlad1=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad2=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad3=zeros(length(vids), length(tVLAD), 'like', tVLAD);

parpool(9);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
parfor i=1:length(pathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
   % desc = NormalizeRowsUnit(desc);
   
    
    
    vlad1(i, :)=VLAD_1_mean(desc, vocabulary);
    vlad2(i, :)=comb_percentage_minVocab_VLAD_1(desc, vocabulary, 0.5)
    vlad3(i, :)=comb_percentage_minVocab_VLAD_1(desc, vocabulary, 1)
    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=2;
allDist=cell(1, nEncoding);

n_vlad1=NormalizeRowsUnit(PowerNormalization(vlad1, 0.5));
allDist{1}=n_vlad1 * n_vlad1';
clear n_vlad1

n_vlad2=NormalizeRowsUnit(PowerNormalization(vlad2, 0.5));
allDist{2}=n_vlad2 * n_vlad2';
clear n_vlad2

n_vlad3=NormalizeRowsUnit(PowerNormalization(vlad3, 0.5));
allDist{3}=n_vlad3 * n_vlad3';
clear n_vlad3


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

saveName = ['/home/ionut/Data/results/ICPR2016_rezults/' 'clfsOut/'  DescParam2Name(descParam) '_VLAD_.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = ['/home/ionut/Data/results/ICPR2016_rezults/' 'videoRep/' DescParam2Name(descParam) '_VLAD_.mat'];
 save(saveName2, '-v7.3', 'vlad1', 'vlad2', 'vlad3');



mean(mean(cat(2, all_accuracy{1}{:})))
mean(mean(cat(2, all_accuracy{2}{:})))


