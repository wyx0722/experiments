global DATAopts;
DATAopts = HMDB51Init;
addpath('./../..')
addpath('./..')

[allVids, labs, splits] = GetVideosPlusLabels();

alpha_handF=0.1
alpha_deepF=0.5


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/clsfOut/' 'FEVid_IDTClusters64_128_256_512_DatasetHMBD51IDTfeatureHOF_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim54spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
hof_clsfOut=all_clfsOut;



pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/clsfOut/' 'FEVid_IDTClusters64_128_256_512_DatasetHMBD51IDTfeatureHOG_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
hog_clsfOut=all_clfsOut;


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/clsfOut/' 'FEVid_IDTClusters64_128_256_512_DatasetHMBD51IDTfeatureMBHx_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
mbhx_clsfOut=all_clfsOut;

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/clsfOut/' 'FEVid_IDTClusters64_128_256_512_DatasetHMBD51IDTfeatureMBHy_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
mbhy_clsfOut=all_clfsOut;


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/clsfOut/' 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMBD51Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19pcaDim256spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
spVGG19_clsfOut=all_clfsOut;

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/clsfOut/' 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMBD51Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit1VGG16pcaDim256spClusters2_4_8_16_32_64_128_256_.mat']
load(pathF);
tempVGG16_clsfOut=all_clfsOut;
clear all_clfsOut

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/hmdb51/clsfOut/' 'clfsOut_ealrlyFusion_v256_v512_spV32.mat']
load(pathF);
ealryFision_all_clfsOut=all_clfsOut;




poz=[9] %!!!!!!!change
nEncoding=length(poz)*6;
mean_all_clfsOut=cell(nEncoding,1);
mean_all_accuracy=cell(nEncoding,1);

lateF_clsfOut=cell(nEncoding,3);
lateF_accuracy=cell(nEncoding,3);

enc=1


maxAcc_iDT=0;
weightsBest_iDT=zeros(1,4);
weights=getSolutions( 0:0.1:1, 4);
meanACC=zeros(1,3);

%late fusion for iDT
fprintf('Start iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI  = splits(:,i) == 2;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = weights(w,1)*hof_clsfOut{poz, i} + weights(w,2)*hog_clsfOut{poz, i} + weights(w,3)*mbhx_clsfOut{poz, i} + weights(w,4)*mbhy_clsfOut{poz, i};      
        meanACC(i)=mean(ClassificationAccuracy(clfsOut, testLabs));      
    end
    
    
    if maxAcc_iDT<mean(meanACC)
        maxAcc_iDT=mean(meanACC);
        weightsBest_iDT=weights(W,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    
end
fprintf('The best MAcc for late fusion iDT: %.3f \n',maxAcc_iDT);
fprintf('The best weights for late fusion iDT: %.2f \n',weightsBest_iDT);




%DOUBLE late fusion for iDT

maxAcc_iDT_d=0;
weightsBest_iDT_d=zeros(1,5);
weights=getSolutions( 0:0.1:1, 5);
meanACC=zeros(1,3);

%late fusion for iDT
fprintf('Start iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI  = splits(:,i) == 2;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = weights(w,1)*hof_clsfOut{poz, i} + weights(w,2)*hog_clsfOut{poz, i} + weights(w,3)*mbhx_clsfOut{poz, i} + weights(w,4)*mbhy_clsfOut{poz, i} ...
                    + weights(w,5)*ealryFision_all_clfsOut{3,i};      
        meanACC(i)=mean(ClassificationAccuracy(clfsOut, testLabs));      
    end
    
    
    if maxAcc_iDT_d<mean(meanACC)
        maxAcc_iDT_d=mean(meanACC);
        weightsBest_iDT_d=weights(W,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    
end
fprintf('The best MAcc for double fusion iDT: %.3f \n',maxAcc_iDT_d);
fprintf('The best weights for double fusion iDT: %.2f \n',weightsBest_iDT_d);


% 
% 
% %late fusion for two-stream
% 
%     for i=1:3
%         
%         trainI = splits(:,i) == 1;
%         testI  = splits(:,i) == 2;
%         trainLabs = labs(trainI,:);
%         testLabs = labs(testI,:);
%         
%         clfsOut = spVGG19_clsfOut{poz, i} + tempVGG16_clsfOut{poz, i}; 
%         accuracy = ClassificationAccuracy(clfsOut, testLabs);
%         fprintf('accuracy: %.3f\n', accuracy);
% 
%         lateF_clsfOut{enc,i}=clfsOut;
%         lateF_accuracy{enc,i}=accuracy;
%     end
%     enc=enc+1
% 
% 
% 
% %DOUBLE late fusion  two-stream
% 
%     for i=1:3
%         
%         trainI = splits(:,i) == 1;
%         testI  = splits(:,i) == 2;
%         trainLabs = labs(trainI,:);
%         testLabs = labs(testI,:);
%         
%         clfsOut = spVGG19_clsfOut{poz, i} + tempVGG16_clsfOut{poz, i} + ealryFision_all_clfsOut{6,i};
%         accuracy = ClassificationAccuracy(clfsOut, testLabs);
%         fprintf('accuracy: %.3f\n', accuracy);
% 
%         lateF_clsfOut{enc,i}=clfsOut;
%         lateF_accuracy{enc,i}=accuracy;
%     end
%     enc=enc+1
% 
% 
% 
% % late fusion for iDT + two-stream 
% 
%     for i=1:3
%         
%         trainI = splits(:,i) == 1;
%         testI  = splits(:,i) == 2;
%         trainLabs = labs(trainI,:);
%         testLabs = labs(testI,:);
%         
%         clfsOut = hof_clsfOut{poz, i} + hog_clsfOut{poz, i} + mbhx_clsfOut{poz, i} + mbhy_clsfOut{poz, i} + spVGG19_clsfOut{poz, i} + tempVGG16_clsfOut{poz, i};
%         accuracy = ClassificationAccuracy(clfsOut, testLabs);
%         fprintf('accuracy: %.3f\n', accuracy);
% 
%         lateF_clsfOut{enc,i}=clfsOut;
%         lateF_accuracy{enc,i}=accuracy;
%     end
%     enc=enc+1
% 
% 
% %DOUBLE late fusion for iDT + two-stream + earlyfusion
% 
%     for i=1:3
%         
%         trainI = splits(:,i) == 1;
%         testI  = splits(:,i) == 2;
%         trainLabs = labs(trainI,:);
%         testLabs = labs(testI,:);
%         
%         clfsOut = hof_clsfOut{poz, i} + hog_clsfOut{poz, i} + mbhx_clsfOut{poz, i} + mbhy_clsfOut{poz, i} + spVGG19_clsfOut{poz, i} + tempVGG16_clsfOut{poz, i} ...
%                     + ealryFision_all_clfsOut{9,i};
%         accuracy = ClassificationAccuracy(clfsOut, testLabs);
%         fprintf('accuracy: %.3f\n', accuracy);
% 
%         lateF_clsfOut{enc,i}=clfsOut;
%         lateF_accuracy{enc,i}=accuracy;
%     end
%     enc=enc+1
% 
% 
% finalAcc=zeros(1,nEncoding);
% for j=1:nEncoding
%     %mean_all_clfsOut{j}=(lateF_clsfOut{j,1} + lateF_clsfOut{j,2} + lateF_clsfOut{j,3})./3;
%     mean_all_accuracy{j}=(lateF_accuracy{j,1} + lateF_accuracy{j,2} + lateF_accuracy{j,3})./3;
%     
%     finalAcc(j)=mean(mean_all_accuracy{j});
%     fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
% end
% 
