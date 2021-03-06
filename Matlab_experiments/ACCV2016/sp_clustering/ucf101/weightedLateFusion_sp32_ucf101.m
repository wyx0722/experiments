global DATAopts;
DATAopts = UCF101Init;
addpath('./../..')
addpath('./..')

[allVids, labs, splits] = GetVideosPlusLabels('Challenge');



pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF101IDTfeatureHOF_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim54spClusters32.mat']
load(pathF);
hof_clsfOut=all_clfsOut;

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF101IDTfeatureHOG_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
hog_clsfOut=all_clfsOut;

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF101IDTfeatureMBHx_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
mbhx_clsfOut=all_clfsOut;

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/clsfOut/' 'FEVid_IDTClusters256_512_DatasetUCF101IDTfeatureMBHy_iTrajMediaTypeIDTNormalisationROOTSIFTpcaDim48spClusters32.mat']
load(pathF);
mbhy_clsfOut=all_clfsOut;


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/clsfOut/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF101Layerpool5MediaTypeDeepFNormalisationNonenetSpVGG19pcaDim256spClusters32.mat']
load(pathF);
spVGG19_clsfOut=all_clfsOut;

tempVGG16_clsfOut=cell(3,3);
pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/clsfOut/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF101Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit1VGG16pcaDim256spClusters32.mat']
load(pathF);
tempVGG16_clsfOut{1,1}=all_clfsOut{1};
tempVGG16_clsfOut{2,1}=all_clfsOut{2};
tempVGG16_clsfOut{3,1}=all_clfsOut{3};


pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/clsfOut/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF101Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit2VGG16pcaDim256spClusters32.mat']
load(pathF);
tempVGG16_clsfOut{1,2}=all_clfsOut{1};
tempVGG16_clsfOut{2,2}=all_clfsOut{2};
tempVGG16_clsfOut{3,2}=all_clfsOut{3};

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/clsfOut/' 'FEVid_deepFeaturesClusters256_512_DatasetUCF101Layerpool5MediaTypeDeepFNormalisationNonenetTempSplit3VGG16pcaDim256spClusters32.mat']
load(pathF);
tempVGG16_clsfOut{1,3}=all_clfsOut{1};
tempVGG16_clsfOut{2,3}=all_clfsOut{2};
tempVGG16_clsfOut{3,3}=all_clfsOut{3};

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf101/clsfOut/' 'earlyFusionClfsOutUCF101.mat']
load(pathF);
ealryFision_all_clfsOut=all_clfsOut;




poz=[3] %!!!!!!!change
nEncoding=length(poz)*6;
mean_all_clfsOut=cell(nEncoding,1);
mean_all_accuracy=cell(nEncoding,1);

lateF_clsfOut=cell(nEncoding,3);
lateF_accuracy=cell(nEncoding,3);

enc=1

interval=0:0.1:1;

maxAcc_iDT=0;
weightsBest_iDT=zeros(1,4);
weights=getSolutions( interval, 4);
meanACC=zeros(1,3);
ld=zeros(1,3);
%late fusion for iDT
fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI=~trainI;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = weights(w,1)*hof_clsfOut{poz, i} + weights(w,2)*hog_clsfOut{poz, i} + weights(w,3)*mbhx_clsfOut{poz, i} + weights(w,4)*mbhy_clsfOut{poz, i};      
        meanACC(i)=mean(ClassificationAccuracy(clfsOut, testLabs));  
        
       if w == 1
          clfsOut = hof_clsfOut{poz, i} + hog_clsfOut{poz, i} + mbhx_clsfOut{poz, i} + mbhy_clsfOut{poz, i}; 
           ld(i)=mean(ClassificationAccuracy(clfsOut, testLabs));  
       end
    end
    
    
    if maxAcc_iDT<mean(meanACC)
        maxAcc_iDT=mean(meanACC);
        weightsBest_iDT=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d=mean(ld)
    end
    
    
end
fprintf('Done!\n');
fprintf('The best MAcc for late fusion iDT: %.3f \n',maxAcc_iDT);
fprintf('The best weights for late fusion iDT: ');
fprintf('%.2f  ',weightsBest_iDT);



%DOUBLE late fusion for iDT

maxAcc_iDT_d=0;
weightsBest_iDT_d=zeros(1,5);
weights=getSolutions( interval, 5);
meanACC=zeros(1,3);


fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI=~trainI;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = weights(w,1)*hof_clsfOut{poz, i} + weights(w,2)*hog_clsfOut{poz, i} + weights(w,3)*mbhx_clsfOut{poz, i} + weights(w,4)*mbhy_clsfOut{poz, i} ...
                    + weights(w,5)*ealryFision_all_clfsOut{3,i};      
        meanACC(i)=mean(ClassificationAccuracy(clfsOut, testLabs));  
        if w == 1
          clfsOut = hof_clsfOut{poz, i} + hog_clsfOut{poz, i} + mbhx_clsfOut{poz, i} + mbhy_clsfOut{poz, i} ...
                    + ealryFision_all_clfsOut{3,i};  
           ld(i)=mean(ClassificationAccuracy(clfsOut, testLabs));  
        end
    end
    
    
    if maxAcc_iDT_d<mean(meanACC)
        maxAcc_iDT_d=mean(meanACC);
        weightsBest_iDT_d=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d2=mean(ld)
    end    
end
fprintf('Done!\n');
fprintf('The best MAcc for double fusion iDT: %.3f \n',maxAcc_iDT_d);
fprintf('The best weights for double fusion iDT: ');
fprintf('%.2f  ',weightsBest_iDT_d);


% 
% 
% %late fusion for two-stream
maxAcc_twoS=0;
weightsBest_twoS=zeros(1,2);
weights=getSolutions( interval, 2);
meanACC=zeros(1,3);

fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI=~trainI;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = weights(w,1)*spVGG19_clsfOut{poz, i} + weights(w,2)*tempVGG16_clsfOut{poz, i};      
        meanACC(i)=mean(ClassificationAccuracy(clfsOut, testLabs));      
       if w == 1
          clfsOut = spVGG19_clsfOut{poz, i} + tempVGG16_clsfOut{poz, i}; 
           ld(i)=mean(ClassificationAccuracy(clfsOut, testLabs));  
       end   
    end
    
    
    if maxAcc_twoS<mean(meanACC)
        maxAcc_twoS=mean(meanACC);
        weightsBest_twoS=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d3=mean(ld)
    end    
end
fprintf('Done!\n');
fprintf('The best MAcc for late fusion two-Stream: %.3f \n',maxAcc_twoS);
fprintf('The best weights for late fusion two-Stream: ');
fprintf('%.2f  ',weightsBest_twoS);


% 
% 
% 
% %DOUBLE late fusion  two-stream
% 
maxAcc_twoS_d=0;
weightsBest_twoS_d=zeros(1,3);
weights=getSolutions( interval, 3);
meanACC=zeros(1,3);

fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI=~trainI;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = weights(w,1)*spVGG19_clsfOut{poz, i} + weights(w,2)*tempVGG16_clsfOut{poz, i} + weights(w,3)*ealryFision_all_clfsOut{6,i};  
        meanACC(i)=mean(ClassificationAccuracy(clfsOut, testLabs));      
       if w == 1
          clfsOut = spVGG19_clsfOut{poz, i} + tempVGG16_clsfOut{poz, i} + ealryFision_all_clfsOut{6,i};  
           ld(i)=mean(ClassificationAccuracy(clfsOut, testLabs));  
       end
    end
    
    
    if maxAcc_twoS_d<mean(meanACC)
        maxAcc_twoS_d=mean(meanACC);
        weightsBest_twoS_d=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d4=mean(ld)
    end    
end
fprintf('Done!\n');
fprintf('The best MAcc for double fusion two-Stream: %.3f \n',maxAcc_twoS_d);
fprintf('The best weights for double fusion two-Stream: ');
fprintf('%.2f  ',weightsBest_twoS_d);
% 
% 
% 
% % late fusion for iDT + two-stream 
% 
maxAcc_iDT_twoS=0;
weightsBest_iDT_twoS=zeros(1,6);
weights=getSolutions( interval, 6);
meanACC=zeros(1,3);

%late fusion for iDT
fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI=~trainI;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = weights(w,1)*hof_clsfOut{poz, i} + weights(w,2)*hog_clsfOut{poz, i} + weights(w,3)*mbhx_clsfOut{poz, i} + weights(w,4)*mbhy_clsfOut{poz, i} ...
            + weights(w,5)*spVGG19_clsfOut{poz, i} + weights(w,6)*tempVGG16_clsfOut{poz, i};      
        meanACC(i)=mean(ClassificationAccuracy(clfsOut, testLabs));      
       if w == 1
          clfsOut = hof_clsfOut{poz, i} + hog_clsfOut{poz, i} + mbhx_clsfOut{poz, i} + mbhy_clsfOut{poz, i} ...
            + spVGG19_clsfOut{poz, i} + tempVGG16_clsfOut{poz, i};  
           ld(i)=mean(ClassificationAccuracy(clfsOut, testLabs));  
       end
    end
    
    
    if maxAcc_iDT_twoS<mean(meanACC)
        maxAcc_iDT_twoS=mean(meanACC);
        weightsBest_iDT_twoS=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d5=mean(ld)
    end    
end
fprintf('Done!\n');
fprintf('The best MAcc for late fusion iDT + twoStream : %.3f \n',maxAcc_iDT_twoS);
fprintf('The best weights for late fusion iDT + twoStream: ');
fprintf('%.2f  ',weightsBest_iDT_twoS);

% 
% 
% %DOUBLE late fusion for iDT + two-stream + earlyfusion
% 
maxAcc_iDT_twoS_d=0;
weightsBest_iDT_twoS_d=zeros(1,7);
weights=getSolutions( interval, 7);
meanACC=zeros(1,3);

%late fusion for iDT
fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI=~trainI;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = weights(w,1)*hof_clsfOut{poz, i} + weights(w,2)*hog_clsfOut{poz, i} + weights(w,3)*mbhx_clsfOut{poz, i} + weights(w,4)*mbhy_clsfOut{poz, i} ...
            + weights(w,5)*spVGG19_clsfOut{poz, i} + weights(w,6)*tempVGG16_clsfOut{poz, i} + weights(w,7)*ealryFision_all_clfsOut{9,i};      
        meanACC(i)=mean(ClassificationAccuracy(clfsOut, testLabs));      
       if w == 1
          clfsOut = hof_clsfOut{poz, i} + hog_clsfOut{poz, i} + mbhx_clsfOut{poz, i} + mbhy_clsfOut{poz, i} ...
            + spVGG19_clsfOut{poz, i} + tempVGG16_clsfOut{poz, i} + ealryFision_all_clfsOut{9,i};   
           ld(i)=mean(ClassificationAccuracy(clfsOut, testLabs));  
       end
    end
    
    
    if maxAcc_iDT_twoS_d<mean(meanACC)
        maxAcc_iDT_twoS_d=mean(meanACC);
        weightsBest_iDT_twoS_d=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d6=mean(ld)
    end    
end
fprintf('Done!\n');
fprintf('The best MAcc for DOUBLE fusion iDT + twoStream : %.3f \n',maxAcc_iDT_twoS_d);
fprintf('The best weights for DOUBLE fusion iDT + twoStream: ');
fprintf('%.2f  ',weightsBest_iDT_twoS_d);