% 
% addpath('./../');%!!!!!!!
% addpath('./../../');%!!!!!!!
% 
% global DATAopts;
% DATAopts = HMDB51Init;
% [allVids, labs, splits] = GetVideosPlusLabels();
% 
% 
% alpha=0.5;
% datasetName='HMDB51'
% 
%  bazeDir='/home/ionut/asustor_ionut/Data/results/cvpr2017/hmdb51/dimPCA/videoRep/'
%  
% % 'descParam', 'sp32cl256pca64', 'sp32cl256pca128', 'sp32cl256pca256', 'sp32cl256pca0', 'v256pca256', 'v256pca0'
%  
%  
% 
% name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetHMDB51Layerconv5bMediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetC3DpcaDim64_128_256_0_spClusters32.mat']
% load(name);
% 
% sp32cl256pca64(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca64(:, end-(32*256) + 1 : end), 0.5);
% C3D_sp32cl256pca64=sp32cl256pca64;
% 
% sp32cl256pca128(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca128(:, end-(32*256) + 1 : end), 0.5);
% C3D_sp32cl256pca128=sp32cl256pca128;
% 
% sp32cl256pca256(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca256(:, end-(32*256) + 1 : end), 0.5);
% C3D_sp32cl256pca256=sp32cl256pca256;
% 
% sp32cl256pca0(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca0(:, end-(32*256) + 1 : end), 0.5);
% C3D_sp32cl256pca0=sp32cl256pca0;
% 
% 
% name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetSpVGG19pcaDim64_128_256_0_spClusters32.mat']
% load(name);
% 
% sp32cl256pca64(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca64(:, end-(32*256) + 1 : end), 0.5);
% SpVGG19_sp32cl256pca64=sp32cl256pca64;
% 
% sp32cl256pca128(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca128(:, end-(32*256) + 1 : end), 0.5);
% SpVGG19_sp32cl256pca128=sp32cl256pca128;
% 
% sp32cl256pca256(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca256(:, end-(32*256) + 1 : end), 0.5);
% SpVGG19_sp32cl256pca256=sp32cl256pca256;
% 
% sp32cl256pca0(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca0(:, end-(32*256) + 1 : end), 0.5);
% SpVGG19_sp32cl256pca0=sp32cl256pca0;
% 
% 
% name=[bazeDir 'FEVid_deepFeaturesClusters256DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetTempSplit1VGG16pcaDim64_128_256_0_spClusters32.mat']
% load(name);
% 
% sp32cl256pca64(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca64(:, end-(32*256) + 1 : end), 0.5);
% TempSplit1VGG16_sp32cl256pca64=sp32cl256pca64;
% 
% sp32cl256pca128(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca128(:, end-(32*256) + 1 : end), 0.5);
% TempSplit1VGG16_sp32cl256pca128=sp32cl256pca128;
% 
% sp32cl256pca256(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca256(:, end-(32*256) + 1 : end), 0.5);
% TempSplit1VGG16_sp32cl256pca256=sp32cl256pca256;
% 
% sp32cl256pca0(:, end-(32*256) + 1 :end)=PowerNormalization(sp32cl256pca0(:, end-(32*256) + 1 : end), 0.5);
% TempSplit1VGG16_sp32cl256pca0=sp32cl256pca0;
% 
% 
% clear sp32cl256pca64 sp32cl256pca128 sp32cl256pca256 sp32cl256pca0
% 
% 
% 
% 
% 
% nEncoding=13;
% allDist=cell(1, nEncoding);
% 
% 
% 
% temp=NormalizeRowsUnit(C3D_sp32cl256pca64); allDist{1}=temp * temp';
% temp=NormalizeRowsUnit(C3D_sp32cl256pca128); allDist{2}=temp * temp';
% temp=NormalizeRowsUnit(C3D_sp32cl256pca256); allDist{3}=temp * temp';
% temp=NormalizeRowsUnit(C3D_sp32cl256pca0); allDist{4}=temp * temp';
% 
% temp=NormalizeRowsUnit(SpVGG19_sp32cl256pca64); allDist{5}=temp * temp';
temp=NormalizeRowsUnit(SpVGG19_sp32cl256pca128); allDist{6}=temp * temp';
% temp=NormalizeRowsUnit(SpVGG19_sp32cl256pca256); allDist{7}=temp * temp';
% temp=NormalizeRowsUnit(SpVGG19_sp32cl256pca0); allDist{8}=temp * temp';
% 
% temp=NormalizeRowsUnit(TempSplit1VGG16_sp32cl256pca64); allDist{9}=temp * temp';
% temp=NormalizeRowsUnit(TempSplit1VGG16_sp32cl256pca128); allDist{10}=temp * temp';
% temp=NormalizeRowsUnit(TempSplit1VGG16_sp32cl256pca256); allDist{11}=temp * temp';
% temp=NormalizeRowsUnit(TempSplit1VGG16_sp32cl256pca0); allDist{12}=temp * temp';
% 
% temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(TempSplit1VGG16_sp32cl256pca0), ...
%     NormalizeRowsUnit(SpVGG19_sp32cl256pca0), NormalizeRowsUnit(C3D_sp32cl256pca0) ));
% allDist{13}=temp * temp';

clear temp

% %each row for the cell represents the results for all 3 splits
% all_clfsOut=cell(1,nEncoding);
% all_accuracy=cell(1,nEncoding);
% clfsOut=cell(1,nEncoding);
% accuracy=cell(1,nEncoding);
% %mean_all_clfsOut=cell(nEncoding,1);
% mean_all_accuracy=cell(nEncoding,1);
% 
% cRange = 100;
% nReps = 1;
% nFolds = 3;


parpool(3);



%%%
for k=6:nEncoding %!!!!!!!!!!!!
    k
    parfor i=1:3
        
        trainI = splits(:,i) == 1;
        
       if ~isempty(strfind(datasetName, 'HMDB51'))
            testI  = splits(:,i) == 2;
       elseif ~isempty(strfind(datasetName, 'UCF101'))
            testI=~trainI;
       end
       
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        trainDist = allDist{k}(trainI, trainI);
        testDist = allDist{k}(testI, trainI);
        

        [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
        %fprintf('accuracy: %.3f\n', accuracy);
    end
     all_clfsOut{k}=clfsOut;
     all_accuracy{k}=accuracy;
end

delete(gcp('nocreate'))
%%%%
clear allDist

finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end

 bazeSavePath='/home/ionut/asustor_ionut/Data/results/cvpr2017/hmdb51/dimPCA/';
 
clfsOut_C3D_sp32cl256pca0 = all_clfsOut(4);
acc_C3D_sp32cl256pca0 = all_accuracy(4);
intructions_compute_acc='mean((acc_C3D_sp32cl256pca0{1}{1} + acc_C3D_sp32cl256pca0{1}{2} + acc_C3D_sp32cl256pca0{1}{3})./3)';
saveName = [bazeSavePath 'clfsOut/' 'PNL2memb__C3D_sp32cl256pca0.mat']
save(saveName, '-v7.3', 'clfsOut_C3D_sp32cl256pca0', 'acc_C3D_sp32cl256pca0', 'intructions_compute_acc');

clfsOut_SpVGG19_sp32cl256pca0 = all_clfsOut(8);
acc_SpVGG19_sp32cl256pca0 = all_accuracy(8);
intructions_compute_acc='mean((acc_SpVGG19_sp32cl256pca0{1}{1} + acc_SpVGG19_sp32cl256pca0{1}{2} + acc_SpVGG19_sp32cl256pca0{1}{3})./3)';
saveName = [bazeSavePath 'clfsOut/' 'PNL2memb__SpVGG19_sp32cl256pca0.mat']
save(saveName, '-v7.3', 'clfsOut_SpVGG19_sp32cl256pca0', 'acc_SpVGG19_sp32cl256pca0', 'intructions_compute_acc');

clfsOut_TempSplit1VGG16_sp32cl256pca0 = all_clfsOut(12);
acc_TempSplit1VGG16_sp32cl256pca0 = all_accuracy(12);
intructions_compute_acc='mean((acc_TempSplit1VGG16_sp32cl256pca0{1}{1} + acc_TempSplit1VGG16_sp32cl256pca0{1}{2} + acc_TempSplit1VGG16_sp32cl256pca0{1}{3})./3)';
saveName = [bazeSavePath 'clfsOut/' 'PNL2memb__TempSplit1VGG16_sp32cl256pca0.mat']
save(saveName, '-v7.3', 'clfsOut_TempSplit1VGG16_sp32cl256pca0', 'acc_TempSplit1VGG16_sp32cl256pca0', 'intructions_compute_acc');

 
saveName = [bazeSavePath 'clfsOut/' 'PNL2memb_all__classification_with_memb_norm.mat']
save(saveName, '-v7.3', 'all_clfsOut', 'all_accuracy');

 
