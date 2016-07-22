global DATAopts;
DATAopts = HMDB51Init;
addpath('./../..')
addpath('./..')

[allVids, labs, splits] = GetVideosPlusLabels();


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

%late fusion for iDT
for k=1:length(poz)
    k
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI  = splits(:,i) == 2;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = hof_clsfOut{poz(k), i} + hog_clsfOut{poz(k), i} + mbhx_clsfOut{poz(k), i} + mbhy_clsfOut{poz(k), i};
        accuracy = ClassificationAccuracy(clfsOut, testLabs);
        fprintf('accuracy: %.3f\n', accuracy);

        lateF_clsfOut{enc,i}=clfsOut;
        lateF_accuracy{enc,i}=accuracy;
    end
    enc=enc+1
end

%DOUBLE late fusion for iDT
for k=1:length(poz)
    k
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI  = splits(:,i) == 2;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = hof_clsfOut{poz(k), i} + hog_clsfOut{poz(k), i} + mbhx_clsfOut{poz(k), i} + mbhy_clsfOut{poz(k), i} + ealryFision_all_clfsOut{3,i};
        accuracy = ClassificationAccuracy(clfsOut, testLabs);
        fprintf('accuracy: %.3f\n', accuracy);

        lateF_clsfOut{enc,i}=clfsOut;
        lateF_accuracy{enc,i}=accuracy;
    end
    enc=enc+1
end



%late fusion for two-stream
for k=1:length(poz)
    k
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI  = splits(:,i) == 2;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = spVGG19_clsfOut{poz(k), i} + tempVGG16_clsfOut{poz(k), i}; 
        accuracy = ClassificationAccuracy(clfsOut, testLabs);
        fprintf('accuracy: %.3f\n', accuracy);

        lateF_clsfOut{enc,i}=clfsOut;
        lateF_accuracy{enc,i}=accuracy;
    end
    enc=enc+1
end


%DOUBLE late fusion  two-stream
for k=1:length(poz)
    k
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI  = splits(:,i) == 2;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = spVGG19_clsfOut{poz(k), i} + tempVGG16_clsfOut{poz(k), i} + ealryFision_all_clfsOut{6,i};
        accuracy = ClassificationAccuracy(clfsOut, testLabs);
        fprintf('accuracy: %.3f\n', accuracy);

        lateF_clsfOut{enc,i}=clfsOut;
        lateF_accuracy{enc,i}=accuracy;
    end
    enc=enc+1
end


% late fusion for iDT + two-stream 
for k=1:length(poz)
    k
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI  = splits(:,i) == 2;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = hof_clsfOut{poz(k), i} + hog_clsfOut{poz(k), i} + mbhx_clsfOut{poz(k), i} + mbhy_clsfOut{poz(k), i} + spVGG19_clsfOut{poz(k), i} + tempVGG16_clsfOut{poz(k), i};
        accuracy = ClassificationAccuracy(clfsOut, testLabs);
        fprintf('accuracy: %.3f\n', accuracy);

        lateF_clsfOut{enc,i}=clfsOut;
        lateF_accuracy{enc,i}=accuracy;
    end
    enc=enc+1
end

%DOUBLE late fusion for iDT + two-stream + earlyfusion
for k=1:length(poz)
    k
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI  = splits(:,i) == 2;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        clfsOut = hof_clsfOut{poz(k), i} + hog_clsfOut{poz(k), i} + mbhx_clsfOut{poz(k), i} + mbhy_clsfOut{poz(k), i} + spVGG19_clsfOut{poz(k), i} + tempVGG16_clsfOut{poz(k), i} ...
                    + ealryFision_all_clfsOut{9,i};
        accuracy = ClassificationAccuracy(clfsOut, testLabs);
        fprintf('accuracy: %.3f\n', accuracy);

        lateF_clsfOut{enc,i}=clfsOut;
        lateF_accuracy{enc,i}=accuracy;
    end
    enc=enc+1
end

finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(lateF_clsfOut{j,1} + lateF_clsfOut{j,2} + lateF_clsfOut{j,3})./3;
    mean_all_accuracy{j}=(lateF_accuracy{j,1} + lateF_accuracy{j,2} + lateF_accuracy{j,3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end

