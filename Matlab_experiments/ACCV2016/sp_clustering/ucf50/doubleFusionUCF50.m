% global DATAopts;
% DATAopts = UCFInit;
% addpath('./../..')
% addpath('./..')
% 
% [vids, labs, groups] = GetVideosPlusLabels('Full');
% 
% 
% pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureHOF_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim54spClusters32.mat']
% load(pathF);
% hof_clsfOut=all_clfsOut;
% 
% pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureHOG_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
% load(pathF);
% hog_clsfOut=all_clfsOut;
% 
% pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureMBHx_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
% load(pathF);
% mbhx_clsfOut=all_clfsOut;
% 
% pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF50IDTfeatureMBHy_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
% load(pathF);
% mbhy_clsfOut=all_clfsOut;
% 
% 
% pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF50Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19pcaDim256spClusters32.mat']
% load(pathF);
% spVGG19_clsfOut=all_clfsOut;
% 
% pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF50Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit1VGG16pcaDim256spClusters32.mat']
% load(pathF);
% tempVGG16_clsfOut=all_clfsOut;
% 
% pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'clfsOutEarlyFusionUCF50.mat']
% load(pathF);
% earlyFusion_clsfOut=all_clfsOut;


%% Do classification
nEncoding=3;%!!!!!!!!change!!!!!!


lateF_all_clfsOut=cell(1,nEncoding);
lateF_all_accuracy=cell(1,nEncoding);

enc=1;


w1=0.25/4
w2=0.25/2
w3=0.5


%late fusion for iDT + two-stream + earlyFusionOutput
for k=1:3

% 
% Leave-one-group-out cross-validation
for i=1:max(groups)
    testI = groups == i;
    trainI = ~testI;
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
     clfsOut{i} = w1*hof_clsfOut{k}{i} + w1*hog_clsfOut{k}{i} +w1* mbhx_clsfOut{k}{i} + w1*mbhy_clsfOut{k}{i} + w2*spVGG19_clsfOut{k}{i} + w2*tempVGG16_clsfOut{k}{i} ...
         + w3*earlyFusion_clsfOut{k}{i};
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