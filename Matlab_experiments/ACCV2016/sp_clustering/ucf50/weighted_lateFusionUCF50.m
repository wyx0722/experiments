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

pathF=['/home/ionut/asustor_ionut_2/Data/results/mmm2016/ucf50/clsfOut/' 'clfsOutEarlyFusionUCF50.mat']
load(pathF);
ealryFision_all_clfsOut=all_clfsOut;


%% Do classification
nEncoding=9;%!!!!!!!!change!!!!!!


lateF_all_clfsOut=cell(1,nEncoding);
lateF_all_accuracy=cell(1,nEncoding);

enc=1;

poz=[3] %!!!!!!!change
interval=0:0.1:1;



maxAcc_iDT=0;
weightsBest_iDT=zeros(1,4);
weights=getSolutions( interval, 4);

 
%late fusion for iDT
fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);
        
        clfsOut{i} = weights(w,1)*hof_clsfOut{poz}{i} + weights(w,2)*hog_clsfOut{poz}{i} + weights(w,3)*mbhx_clsfOut{poz}{i} + weights(w,4)*mbhy_clsfOut{poz}{i};      
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);  
        
       if w == 1
          clfsOut_noW = hof_clsfOut{poz}{i} + hog_clsfOut{poz}{i} + mbhx_clsfOut{poz}{i} + mbhy_clsfOut{poz}{i}; 
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    if maxAcc_iDT<mean(mean(cat(2, acc{:})));
        maxAcc_iDT=mean(mean(cat(2, acc{:})));;
        weightsBest_iDT=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d=mean(mean(cat(2, ld{:})));
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



fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);
        
        clfsOut{i} = weights(w,1)*hof_clsfOut{poz}{i} + weights(w,2)*hog_clsfOut{poz}{i} + weights(w,3)*mbhx_clsfOut{poz}{i} + weights(w,4)*mbhy_clsfOut{poz}{i} ...
                    + weights(w,5)*ealryFision_all_clfsOut{3}{i};      
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);  
        if w == 1
          clfsOut_noW = hof_clsfOut{poz}{i} + hog_clsfOut{poz}{i} + mbhx_clsfOut{poz}{i} + mbhy_clsfOut{poz}{i} ...
                    + ealryFision_all_clfsOut{3}{i};  
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
        end
    end
    
    
    if maxAcc_iDT_d<mean(mean(cat(2, acc{:})));
        maxAcc_iDT_d=mean(mean(cat(2, acc{:})));;
        weightsBest_iDT_d=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d2=mean(mean(cat(2, ld{:})));
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


fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);
        
        clfsOut{i} = weights(w,1)*spVGG19_clsfOut{poz}{i} + weights(w,2)*tempVGG16_clsfOut{poz}{i};      
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);      
       if w == 1
          clfsOut_noW = spVGG19_clsfOut{poz}{i} + tempVGG16_clsfOut{poz}{i}; 
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end   
    end
    
    
    if maxAcc_twoS<mean(mean(cat(2, acc{:})));
        maxAcc_twoS=mean(mean(cat(2, acc{:})));
        weightsBest_twoS=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d3=mean(mean(cat(2, ld{:})));
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


fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);
        
        clfsOut{i} = weights(w,1)*spVGG19_clsfOut{poz}{i} + weights(w,2)*tempVGG16_clsfOut{poz}{i} + weights(w,3)*ealryFision_all_clfsOut{6}{i};  
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);      
       if w == 1
          clfsOut_noW = spVGG19_clsfOut{poz}{i} + tempVGG16_clsfOut{poz}{i} + ealryFision_all_clfsOut{6}{i};  
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    if maxAcc_twoS_d<mean(mean(cat(2, acc{:})));
        maxAcc_twoS_d=mean(mean(cat(2, acc{:})));;
        weightsBest_twoS_d=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d4=mean(mean(cat(2, ld{:})));
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


%late fusion for iDT
fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);
        
        clfsOut{i} = weights(w,1)*hof_clsfOut{poz}{i} + weights(w,2)*hog_clsfOut{poz}{i} + weights(w,3)*mbhx_clsfOut{poz}{i} + weights(w,4)*mbhy_clsfOut{poz}{i} ...
            + weights(w,5)*spVGG19_clsfOut{poz}{i} + weights(w,6)*tempVGG16_clsfOut{poz}{i};      
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);      
       if w == 1
          clfsOut_noW = hof_clsfOut{poz}{i} + hog_clsfOut{poz}{i} + mbhx_clsfOut{poz}{i} + mbhy_clsfOut{poz}{i} ...
            + spVGG19_clsfOut{poz}{i} + tempVGG16_clsfOut{poz}{i};  
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    if maxAcc_iDT_twoS<mean(mean(cat(2, acc{:})));
        maxAcc_iDT_twoS=mean(mean(cat(2, acc{:})));;
        weightsBest_iDT_twoS=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d5=mean(mean(cat(2, ld{:})));
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


%late fusion for iDT
fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);
        
        clfsOut{i} = weights(w,1)*hof_clsfOut{poz}{i} + weights(w,2)*hog_clsfOut{poz}{i} + weights(w,3)*mbhx_clsfOut{poz}{i} + weights(w,4)*mbhy_clsfOut{poz}{i} ...
            + weights(w,5)*spVGG19_clsfOut{poz}{i} + weights(w,6)*tempVGG16_clsfOut{poz}{i} + weights(w,7)*ealryFision_all_clfsOut{9}{i};      
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);      
       if w == 1
          clfsOut_noW = hof_clsfOut{poz}{i} + hog_clsfOut{poz}{i} + mbhx_clsfOut{poz}{i} + mbhy_clsfOut{poz}{i} ...
            + spVGG19_clsfOut{poz}{i} + tempVGG16_clsfOut{poz}{i} + ealryFision_all_clfsOut{9}{i};   
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    if maxAcc_iDT_twoS_d<mean(mean(cat(2, acc{:})));
        maxAcc_iDT_twoS_d=mean(mean(cat(2, acc{:})));;
        weightsBest_iDT_twoS_d=weights(w,:);   
    end
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        d6=mean(mean(cat(2, ld{:})));
    end    
end
fprintf('Done!\n');
fprintf('The best MAcc for DOUBLE fusion iDT + twoStream : %.3f \n',maxAcc_iDT_twoS_d);
fprintf('The best weights for DOUBLE fusion iDT + twoStream: ');
fprintf('%.2f  ',weightsBest_iDT_twoS_d);