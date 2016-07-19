global DATAopts;
DATAopts = UCFInit;
addpath('./../..')
addpath('./..')

[vids, labs, groups] = GetVideosPlusLabels('Full');


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureHOF_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim54spClusters32.mat']
load(pathF);
hof_clsfOut=all_clfsOut;

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureHOG_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
hog_clsfOut=all_clfsOut;

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureMBHx_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
mbhx_clsfOut=all_clfsOut;

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureMBHy_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
mbhy_clsfOut=all_clfsOut;


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF50Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19pcaDim256spClusters32.mat']
load(pathF);
spVGG19_clsfOut=all_clfsOut;

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF50Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit1VGG16pcaDim256spClusters32.mat']
load(pathF);
tempVGG16_clsfOut=all_clfsOut;



%% Do classification
nEncoding=9;%!!!!!!!!change!!!!!!


lateF_all_clfsOut=cell(1,nEncoding);
lateF_all_accuracy=cell(1,nEncoding);

enc=1;

%late fusion for iDT
for k=1:3

% 
% Leave-one-group-out cross-validation
for i=1:max(groups)
    testI = groups == i;
    trainI = ~testI;
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    clfsOut{i} = hof_clsfOut{k}{i} + hog_clsfOut{k}{i} + mbhx_clsfOut{k}{i} + mbhy_clsfOut{k}{i};
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

lateF_all_clfsOut{enc}=clfsOut;
lateF_all_accuracy{enc}=accuracy;
enc=enc+1;
k
perGroupAccuracy = mean(cat(2, accuracy{:}))'

mean(perGroupAccuracy)

end


%late fusion for two-stream
for k=1:3

% 
% Leave-one-group-out cross-validation
for i=1:max(groups)
    testI = groups == i;
    trainI = ~testI;
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
     clfsOut{i} = spVGG19_clsfOut{k}{i} + tempVGG16_clsfOut{k}{i};
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

lateF_all_clfsOut{enc}=clfsOut;
lateF_all_accuracy{enc}=accuracy;
enc=enc+1;
k
perGroupAccuracy = mean(cat(2, accuracy{:}))'

mean(perGroupAccuracy)

end


%late fusion for iDT + two-stream
for k=1:3

% 
% Leave-one-group-out cross-validation
for i=1:max(groups)
    testI = groups == i;
    trainI = ~testI;
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
     clfsOut{i} = spVGG19_clsfOut{k}{i} + tempVGG16_clsfOut{k}{i} + hof_clsfOut{k}{i} + hog_clsfOut{k}{i} + mbhx_clsfOut{k}{i} + mbhy_clsfOut{k}{i};
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

lateF_all_clfsOut{enc}=clfsOut;
lateF_all_accuracy{enc}=accuracy;
enc=enc+1;
k
perGroupAccuracy = mean(cat(2, accuracy{:}))'

mean(perGroupAccuracy)

end







finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    finalAcc(j)=mean(mean(cat(2, lateF_all_accuracy{j}{:}), 2));
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));  
end