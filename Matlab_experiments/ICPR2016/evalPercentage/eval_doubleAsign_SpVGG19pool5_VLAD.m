
global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='SpVGG19';
descParam.Normalisation='ROOTSIFT';

descParam.numClusters = 256;

descParam.pcaDim = 256;

descParam

%%%%%%%%%%
bazePathFeatures='/home/ionut/halley_ionut/Data/VGG_19_v-features_rawFrames_UCF50/Videos/' %change


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
    
vlad0=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad01=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad02=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad03=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad04=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad05=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad06=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad07=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad08=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad09=zeros(length(vids), length(tVLAD), 'like', tVLAD);
vlad1=zeros(length(vids), length(tVLAD), 'like', tVLAD);


parpool(13);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
parfor i=1:length(pathFeatures)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{i}, descParam, pcaMap);
   % desc = NormalizeRowsUnit(desc);
   
    
    
   
    vlad0(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 0);
    vlad01(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 0.1);
    vlad02(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 0.2);
    vlad03(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 0.3);
    vlad04(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 0.4);
    vlad05(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 0.5);
    vlad06(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 0.6);
    vlad07(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 0.7);
    vlad08(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 0.8);
    vlad09(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 0.9);
    vlad1(i, :)=doubleAssign_VLAD_1(desc, vocabulary, 1);
 

    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=11;
allDist=cell(1, nEncoding);

n_vlad0=NormalizeRowsUnit(PowerNormalization(vlad0, 0.5));
allDist{1}=n_vlad0 * n_vlad0';
clear n_vlad0

n_vlad01=NormalizeRowsUnit(PowerNormalization(vlad01, 0.5));
allDist{2}=n_vlad01 * n_vlad01';
clear n_vlad01

n_vlad02=NormalizeRowsUnit(PowerNormalization(vlad02, 0.5));
allDist{3}=n_vlad02 * n_vlad02';
clear n_vlad02

n_vlad03=NormalizeRowsUnit(PowerNormalization(vlad03, 0.5));
allDist{4}=n_vlad03 * n_vlad03';
clear n_vlad03

n_vlad04=NormalizeRowsUnit(PowerNormalization(vlad04, 0.5));
allDist{5}=n_vlad04 * n_vlad04';
clear n_vlad04

n_vlad05=NormalizeRowsUnit(PowerNormalization(vlad05, 0.5));
allDist{6}=n_vlad05 * n_vlad05';
clear n_vlad05

n_vlad06=NormalizeRowsUnit(PowerNormalization(vlad06, 0.5));
allDist{7}=n_vlad06 * n_vlad06';
clear n_vlad06

n_vlad07=NormalizeRowsUnit(PowerNormalization(vlad07, 0.5));
allDist{8}=n_vlad07 * n_vlad07';
clear n_vlad07

n_vlad08=NormalizeRowsUnit(PowerNormalization(vlad08, 0.5));
allDist{9}=n_vlad08 * n_vlad08';
clear n_vlad08

n_vlad09=NormalizeRowsUnit(PowerNormalization(vlad09, 0.5));
allDist{10}=n_vlad09 * n_vlad09';
clear n_vlad09

n_vlad1=NormalizeRowsUnit(PowerNormalization(vlad1, 0.5));
allDist{11}=n_vlad1 * n_vlad1';
clear n_vlad1




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


fileName=sprintf('/home/ionut/experiments/Matlab_experiments/ICPR2016/results/evalDoubleAssign/results_UCF50_evalDoubleAssign_Features%s_Layer%s_Network%s_Clusters%d_PCAdim%d_norm%s__VLAD.txt', descParam.MediaType,descParam.Layer, descParam.net,descParam.numClusters,descParam.pcaDim, descParam.Normalisation); 
fileID=fopen(fileName, 'a');
    
p=0:0.1:1;
for k=1:nEncoding
    acc=mean(mean(cat(2, all_accuracy{k}{:})));
    fprintf(fileID, 'doble assignemnt VLAD with keeping the percentage  %.1f --> acc= %.4f \r\n', p(k), acc);   
end
fclose(fileID);



