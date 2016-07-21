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

maxAcc_iDT=zeros(1,3);
weightsBest_iDT=zeros(3,4);
weights=getSolutions( 0:0.1:1, 4);

%late fusion for iDT
for w=1:size(weights, 1)
    for k=1:3

        % 
        % Leave-one-group-out cross-validation
        for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);

            clfsOut{i} = weights(w,1)*hof_clsfOut{k}{i} + weights(w,2)*hog_clsfOut{k}{i} + weights(w,3)*mbhx_clsfOut{k}{i} + weights(w,4)*mbhy_clsfOut{k}{i};
            accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);

        end
        
        acc=mean(mean(cat(2, accuracy{:})));
        %fprintf('%d: accuracy: %.3f\n', i, acc);
        
        if acc>maxAcc_iDT(k)
            maxAcc_iDT(k)=acc;
            weightsBest_iDT(k, :)=weights(w, :);           
        end
%         lateF_all_clfsOut{enc}=clfsOut;
%         lateF_all_accuracy{enc}=accuracy;
%         enc=enc+1;
%         k
%         perGroupAccuracy = mean(cat(2, accuracy{:}))'
% 
%         mean(perGroupAccuracy)

    end
end


fprintf('Encoding --> MAcc: %.3f \n', maxAcc_iDT); 
weightsBest_iDT

