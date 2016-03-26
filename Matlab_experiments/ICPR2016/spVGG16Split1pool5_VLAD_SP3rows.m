
global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='SpSplit1VGG16';
descParam.Normalisation='ROOTSIFT';


descParam.numClusters = 256;

descParam.pcaDim = 256;

sRow=3;

descParam

%%%%%%%%%%
bazePathFeatures='/home/ionut/halley_ionut/Data/spatial_vgg_16_split1_rawFrames_UCF50/Videos/'; %change


vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '/pool5.txt'];
end


                                            
[vocabulary, pcaMap] = CreateVocabularyKmeansPca(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim); 
                                            
%vocabulary = NormalizeRowsUnit(vocabulary); %make unit length

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i)) '/pool5.txt'];
end



    [tDesc] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
    %tDesc=NormalizeRowsUnit(tDesc);
    tVLAD=VLAD_1_mean(tDesc, vocabulary);
    
vlad1_1=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad1_2=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad1_3=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad1_4=zeros(length(vids), length(tVLAD), 'like', tVLAD);

vlad2_1=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad2_2=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad2_3=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad2_4=zeros(length(vids), length(tVLAD), 'like', tVLAD);

parpool(5);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
parfor i=1:length(pathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
   % desc = NormalizeRowsUnit(desc);
   
    
    [idx] = SpatialPyramid_RowsMaps(info.spinfo, sRow)
    
    
    
    vlad1_1(i, :)=VLAD_1_mean(desc(idx(:, 1), :), vocabulary);
    vlad1_2(i, :)=VLAD_1_mean(desc(idx(:, 2), :), vocabulary);
    vlad1_3(i, :)=VLAD_1_mean(desc(idx(:, 3), :), vocabulary);
    vlad1_4(i, :)=VLAD_1_mean(desc(idx(:, 4), :), vocabulary);
    
    vlad2_1(i, :)=doubleAssign_VLAD_1(desc(idx(:, 1), :), vocabulary, 1);
    vlad2_2(i, :)=doubleAssign_VLAD_1(desc(idx(:, 2), :), vocabulary, 1);
    vlad2_3(i, :)=doubleAssign_VLAD_1(desc(idx(:, 3), :), vocabulary, 1);
    vlad2_4(i, :)=doubleAssign_VLAD_1(desc(idx(:, 4), :), vocabulary, 1);
    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');

vlad1=cat(2,vlad1_1, vlad1_2, vlad1_3, vlad1_4);
vlad2=cat(2,vlad2_1, vlad2_2, vlad2_3, vlad2_4);

%% Do classification

nEncoding=2;
allDist=cell(1, nEncoding);

n_vlad1=NormalizeRowsUnit(PowerNormalization(vlad1, 0.5));
allDist{1}=n_vlad1 * n_vlad1';
clear n_vlad1

n_vlad2=NormalizeRowsUnit(PowerNormalization(vlad2, 0.5));
allDist{2}=n_vlad2 * n_vlad2';
clear n_vlad2


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




mean(mean(cat(2, all_accuracy{1}{:})))
mean(mean(cat(2, all_accuracy{2}{:})))


