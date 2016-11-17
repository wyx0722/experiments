
addpath('./../');%!!!!!!!
addpath('./../../');%!!!!!!!

global DATAopts;
DATAopts = UCFInit;
[vids, labs, groups] = GetVideosPlusLabels('Full');


bazeDir='/home/ionut/asustor_ionut/Data/results/cvpr2017/forFusion_state_of_the_art/ucf50/clfsOut/'

name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF50Layerconv5bMediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netC3DpcaDim256_0_spClusters32_64_sp_clDim_check32_64__PNL2memb__sp32cl256pca0.mat']
load(name);
c3d=clfsOut_sp32cl256pca0(1); clear clfsOut_sp32cl256pca0

name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF50Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256netSpVGG19pcaDim256_0_spClusters32_64_sp_clDim_check32_64__PNL2memb__sp32cl256pca0.mat']
load(name);
scn=clfsOut_sp32cl256pca0(1); clear clfsOut_sp32cl256pca0

name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetUCF50Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonecheck_Clusters256_256_gmmSize256nettempVGG16pcaDim256_0_spClusters32_64_sp_clDim_check32_64__PNL2memb__sp32cl256pca0.mat']
load(name);
tcn=clfsOut_sp32cl256pca0(1); clear clfsOut_sp32cl256pca0



name=[bazeDir 'FEVidHmgDenseBlockSize8_8_6_DatasetUCF50FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72_fisherAll__all_accuracy.mat']
load(name);
hmg=all_clfsOut(1); clear all_clfsOut

name=[bazeDir 'FEVid_IDTDatasetUCF50IDTfeatureHOFMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim54_fisherAll__all_accuracy.mat']
load(name);
hof=all_clfsOut(1); clear all_clfsOut

name=[bazeDir 'FEVid_IDTDatasetUCF50IDTfeatureHOGMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48_fisherAll__all_accuracy.mat']
load(name);
hog=all_clfsOut(1); clear all_clfsOut

name=[bazeDir 'FEVid_IDTDatasetUCF50IDTfeatureMBHxMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48_fisherAll__all_accuracy.mat']
load(name);
mbhx=all_clfsOut(1); clear all_clfsOut

name=[bazeDir 'FEVid_IDTDatasetUCF50IDTfeatureMBHyMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48_fisherAll__all_accuracy.mat']
load(name);
mbhy=all_clfsOut(1); clear all_clfsOut


name=[bazeDir 'all_early_fusion.mat']
load(name);
df=all_clfsOut(1);
df_hmg=all_clfsOut(4);
df_idt=all_clfsOut(5);
df_idt_hmg=all_clfsOut(6);


load('/home/ionut/asustor_ionut/Data/results/cvpr2017/forFusion_state_of_the_art/weightsCombinations_forLateDouble_fusion_interval005.mat');

%interval=0:0.1:1;


%~~~~~~Deep features: SCN+TCN+C3D

%wighted late
weights=comb3;

L_weights_df=weights;
L_mean_acc_df=zeros(size(weights,1),1);


fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);  
        
        clfsOut{i} = weights(w,1)*c3d{1}{i} + weights(w,2)*scn{1}{i} + weights(w,3)*tcn{1}{i};
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);  
       if w == 1
          clfsOut_noW =  c3d{1}{i} + scn{1}{i} + tcn{1}{i};
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    L_mean_acc_df(w)=mean(mean(cat(2, acc{:})));
    
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        sumLate_df=mean(mean(cat(2, ld{:})))
    end
    
    
end
   
%wighted Double
weights=comb4;

D_weights_df=weights;
D_mean_acc_df=zeros(size(weights,1),1);

fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);  
        
        clfsOut{i} = weights(w,1)*c3d{1}{i} + weights(w,2)*scn{1}{i} + weights(w,3)*tcn{1}{i} + weights(w,4)*df{1}{i};
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);  
       if w == 1
          clfsOut_noW =  c3d{1}{i} + scn{1}{i} + tcn{1}{i} + df{1}{i};
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    D_mean_acc_df(w)=mean(mean(cat(2, acc{:})));
    
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        sumDouble_df=mean(mean(cat(2, ld{:})))
    end
    
    
end
%~~~~~~~~~~~~~~~~~~






%~~~~~~Deep features +HMG: SCN+TCN+C3D+HMG

%wighted late
weights=comb4;

L_weights_df_hmg=weights;
L_mean_acc_df_hmg=zeros(size(weights,1),1);

fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);  
        
        clfsOut{i} = weights(w,1)*c3d{1}{i} + weights(w,2)*scn{1}{i} + weights(w,3)*tcn{1}{i} + weights(w,4)*hmg{1}{i};
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);  
       if w == 1
          clfsOut_noW =  c3d{1}{i} + scn{1}{i} + tcn{1}{i} + hmg{1}{i};
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    L_mean_acc_df_hmg(w)=mean(mean(cat(2, acc{:})));
    
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        sumLate_df_hmg=mean(mean(cat(2, ld{:})))
    end
    
    
end
   
%wighted Double
weights=comb5;

D_weights_df_hmg=weights;
D_mean_acc_df_hmg=zeros(size(weights,1),1);

fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);  
        
        clfsOut{i} = weights(w,1)*c3d{1}{i} + weights(w,2)*scn{1}{i} + weights(w,3)*tcn{1}{i} + weights(w,4)*hmg{1}{i} + weights(w,5)*df_hmg{1}{i};
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);  
       if w == 1
          clfsOut_noW =  c3d{1}{i} + scn{1}{i} + tcn{1}{i} + hmg{1}{i} + df_hmg{1}{i};
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    D_mean_acc_df_hmg(w)=mean(mean(cat(2, acc{:})));
    
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        sumDouble_df_hmg=mean(mean(cat(2, ld{:})))
    end
    
    
end
%~~~~~~~~~~~~~~~~~~



%~~~~~~Deep features +idt: SCN+TCN+C3D+HOG+HOF+MBHx+MBHy

%wighted late
weights=comb7;

L_weights_df_idt=weights;
L_mean_acc_df_idt=zeros(size(weights,1),1);

fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);  
        
        clfsOut{i} = weights(w,1)*c3d{1}{i} + weights(w,2)*scn{1}{i} + weights(w,3)*tcn{1}{i} + weights(w,4)*hof{1}{i} + weights(w,5)*hog{1}{i} + weights(w,6)*mbhx{1}{i} + weights(w,7)*mbhy{1}{i};
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);  
       if w == 1
          clfsOut_noW =  c3d{1}{i} + scn{1}{i} + tcn{1}{i} + hof{1}{i} + hog{1}{i} + mbhx{1}{i} + mbhy{1}{i};
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    L_mean_acc_df_idt(w)=mean(mean(cat(2, acc{:})));
    
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        sumLate_df_idt=mean(mean(cat(2, ld{:})))
    end
    
    
end
   
%wighted Double
weights=comb8;

D_weights_df_idt=weights;
D_mean_acc_df_idt=zeros(size(weights,1),1);

fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);  
        
        clfsOut{i} = weights(w,1)*c3d{1}{i} + weights(w,2)*scn{1}{i} + weights(w,3)*tcn{1}{i} + weights(w,4)*hof{1}{i} + weights(w,5)*hog{1}{i} + weights(w,6)*mbhx{1}{i} + weights(w,7)*mbhy{1}{i} + weights(w,8)*df_idt{1}{i};
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);  
       if w == 1
          clfsOut_noW =  c3d{1}{i} + scn{1}{i} + tcn{1}{i} + hof{1}{i} + hog{1}{i} + mbhx{1}{i} + mbhy{1}{i} + df_idt{1}{i};
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    D_mean_acc_df_idt(w)=mean(mean(cat(2, acc{:})));
    
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        sumDouble_df_idt=mean(mean(cat(2, ld{:})))
    end
    
    
end
%~~~~~~~~~~~~~~~~~~

%~~~~~~Deep features +idt+hmg: SCN+TCN+C3D+HOG+HOF+MBHx+MBHy+HMG

%wighted late
weights=comb8;

L_weights_df_idt_hmg=weights;
L_mean_acc_df_idt_hmg=zeros(size(weights,1),1);

fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);  
        
        clfsOut{i} = weights(w,1)*c3d{1}{i} + weights(w,2)*scn{1}{i} + weights(w,3)*tcn{1}{i} + weights(w,4)*hof{1}{i} + weights(w,5)*hog{1}{i} + weights(w,6)*mbhx{1}{i} + weights(w,7)*mbhy{1}{i} + weights(w,8)*hmg{1}{i};
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);  
       if w == 1
          clfsOut_noW =  c3d{1}{i} + scn{1}{i} + tcn{1}{i} + hof{1}{i} + hog{1}{i} + mbhx{1}{i} + mbhy{1}{i} + hmg{1}{i};
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    L_mean_acc_df_idt_hmg(w)=mean(mean(cat(2, acc{:})));
    
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        sumLate_df_idt_hmg=mean(mean(cat(2, ld{:})))
    end
    
    
end

%wighted Double
weights=comb9;

D_weights_df_idt_hmg=weights;
D_mean_acc_df_idt_hmg=zeros(size(weights,1),1);

fprintf('\n\nStart iterating over: %d \n', size(weights,1));
for w=1:size(weights,1)
    for i=1:max(groups)
            testI = groups == i;
            trainI = ~testI;
            trainLabs = labs(trainI,:);
            testLabs = labs(testI, :);  
        
        clfsOut{i} = weights(w,1)*c3d{1}{i} + weights(w,2)*scn{1}{i} + weights(w,3)*tcn{1}{i} + weights(w,4)*hof{1}{i} + weights(w,5)*hog{1}{i} + weights(w,6)*mbhx{1}{i} + weights(w,7)*mbhy{1}{i} + weights(w,8)*hmg{1}{i} + weights(w,9)*df_idt_hmg{1}{i};
        acc{i}=ClassificationAccuracy(clfsOut{i}, testLabs);  
       if w == 1
          clfsOut_noW =  c3d{1}{i} + scn{1}{i} + tcn{1}{i} + hof{1}{i} + hog{1}{i} + mbhx{1}{i} + mbhy{1}{i} + hmg{1}{i} + df_idt_hmg{1}{i};
           ld{i}=ClassificationAccuracy(clfsOut_noW, testLabs);  
       end
    end
    
    
    D_mean_acc_df_idt_hmg(w)=mean(mean(cat(2, acc{:})));
    
    
    if mod(w,100)==0
        fprintf('%d ', w);
    end
    if w==1
        sumDouble_df_idt_hmg=mean(mean(cat(2, ld{:})))
    end
    
    
end
%~~~~~~~~~~~~~~~~~~


%~~~~~~~df
fprintf('Final results for fusion: c3d + scn + tcn\n')
fprintf('%.3f \n', sumLate_df)
fprintf('%.3f \n', max(L_mean_acc_df))
fprintf('%.3f \n', sumDouble_df)
fprintf('%.3f \n', max(D_mean_acc_df))
[~, idx]=max(L_mean_acc_df);
fprintf('%.2f ', L_weights_df(idx, :))
fprintf('\n');
[~, idx]=max(D_mean_acc_df);
fprintf('%.2f ', D_weights_df(idx, :))
fprintf('\n\n');

%~~~~~~~df+hmg
fprintf('Final results for fusion: c3d + scn + tcn + hmg\n')
fprintf('%.3f \n', sumLate_df_hmg)
fprintf('%.3f \n', max(L_mean_acc_df_hmg))
fprintf('%.3f \n', sumDouble_df_hmg)
fprintf('%.3f \n', max(D_mean_acc_df_hmg))
[~, idx]=max(L_mean_acc_df_hmg);
fprintf('%.2f ', L_weights_df_hmg(idx, :))
fprintf('\n');
[~, idx]=max(D_mean_acc_df_hmg);
fprintf('%.2f ', D_weights_df_hmg(idx, :))
fprintf('\n\n');

%~~~~~~~df+idt
fprintf('Final results for fusion: c3d + scn + tcn + hof + hog + mbhx + mbhy\n')
fprintf('%.3f \n', sumLate_df_idt)
fprintf('%.3f \n', max(L_mean_acc_df_idt))
fprintf('%.3f \n', sumDouble_df_idt)
fprintf('%.3f \n', max(D_mean_acc_df_idt))
[~, idx]=max(L_mean_acc_df_idt);
fprintf('%.2f ', L_weights_df_idt(idx, :))
fprintf('\n');
[~, idx]=max(D_mean_acc_df_idt);
fprintf('%.2f ', D_weights_df_idt(idx, :))
fprintf('\n\n');

%~~~~~~~df+idt + hmg
fprintf('Final results for fusion: c3d + scn + tcn + hof + hog + mbhx + mbhy + hmg\n')
fprintf('%.3f \n', sumLate_df_idt_hmg)
fprintf('%.3f \n', max(L_mean_acc_df_idt_hmg))
fprintf('%.3f \n', sumDouble_df_idt_hmg)
fprintf('%.3f \n', max(D_mean_acc_df_idt_hmg))
[~, idx]=max(L_mean_acc_df_idt_hmg);
fprintf('%.2f ', L_weights_df_idt_hmg(idx, :))
fprintf('\n');
[~, idx]=max(D_mean_acc_df_idt_hmg);
fprintf('%.2f ', D_weights_df_idt_hmg(idx, :))
fprintf('\n\n');