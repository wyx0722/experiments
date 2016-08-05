global DATAopts;
DATAopts = HMDB51Init;


addpath('./..')

clear descParam
descParam.Dataset='HMBD51';
descParam.Split=1;
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='TempSplit1VGG16';
descParam.Normalisation='None'; % L2 or 'ROOTSIFT'
alpha=0.5;%for PN !!!!!!!change!!!!!!!



% switch descParam.MediaType
%     case 'IDT'
%         if strfind(descParam.IDTfeature,'HOF')>0
%             sizeDesc=108;   
%         elseif  strfind(descParam.IDTfeature,'HOG')>0 || strfind(descParam.IDTfeature,'MBHx')>0 || strfind(descParam.IDTfeature,'MBHy')>0  
%             sizeDesc=96;   
%         end
%         descParam.pcaDim = sizeDesc/2;
%     case 'DeepF'
%         descParam.pcaDim=256;%!!!
% end

descParam.pcaDim=0;

descParam.sizeVocab=256;
%the baze path for features
bazePathFeatures='/home/ionut/asustor_ionut_2/Data/hmdb51_action_temporal_vgg_16_split1_features_opticalFlow_tvL1/Videos/'
descParam






[allVids, labs, splits] = GetVideosPlusLabels();



%create the full path of the fetures for each video
allPathFeatures=cell(size(allVids));
for i=1:size(allVids, 1)
    
    if strfind(descParam.MediaType, 'DeepF')>0 
        allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4) '/' descParam.Layer '.txt'];
    elseif strfind(descParam.MediaType, 'IDT')>0 
        allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4)];
    end
end



trainTestSetPathFeatures=allPathFeatures(splits(:, descParam.Split)==1 | splits(:, descParam.Split)==2);%get all the paths for the current split
trainTestSetlabs=labs((splits(:, descParam.Split)==1 | splits(:, descParam.Split)==2), :); %get all the labels for the current split
trainTestSplit=splits((splits(:, descParam.Split)==1 | splits(:, descParam.Split)==2), 1); %get the devision of date between training and testing set for the current split. Exclude the videos not included in the split (0 value)

trainingSetPathFeatures=trainTestSetPathFeatures(trainTestSplit==1); %get the trining set feature paths 
%vocabularyPathFeatures=trainingSetPathFeatures(1:2:end); % build the vocabulary for half of the videos of the training set

trainLabs=trainTestSetlabs(trainTestSplit==1, :);



% 
% all_clfsOut=cell(1,nEncoding);
% all_accuracy=cell(1,nEncoding);
% 
% cRange = 100;
% nReps = 1;
% nFolds = 3;
% 
% parpool(nEncoding);
% parfor k=1:nEncoding
%     k
%     trainI=trainTestSplit==1;
%     testI=~trainI;
%     
%     trainDist = allDist{k}(trainI, trainI);
%     testDist = allDist{k}(testI, trainI);
%     trainLabs = trainTestSetlabs(trainI,:);
%     testLabs = trainTestSetlabs(testI, :);
%     
%     [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
%     accuracy = ClassificationAccuracy(clfsOut, testLabs);
%     fprintf('accuracy: %.3f\n', accuracy);
%     
%     all_clfsOut{k}=clfsOut;
%     all_accuracy{k}=accuracy;
% end
% 
% delete(gcp('nocreate'))
% 
