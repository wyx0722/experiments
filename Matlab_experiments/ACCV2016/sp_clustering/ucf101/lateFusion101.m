global DATAopts;
DATAopts = UCF101Init;
addpath('./../..')
addpath('./..')

[allVids, labs, splits] = GetVideosPlusLabels('Challenge');

alpha_handF=0.1
alpha_deepF=0.5


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






%% Do classification
nEncoding=9;%!!!!!!!!change!!!!!!


mean_all_clfsOut=cell(nEncoding,1);
mean_all_accuracy=cell(nEncoding,1);

lateF_clsfOut=cell(nEncoding,3);
lateF_accuracy=cell(nEncoding,3);


cRange = 100;
nReps = 1;
nFolds = 3;


enc=1;



for k=1:3
    k
    for i=1:3
        trainI=splits(:,i)==1;
        testI=~trainI;

        %trainDist = allDist{k,i}(trainI, trainI);
        %testDist = allDist{k,i}(testI, trainI);
        trainLabs = labs(trainI,:);
        testLabs = labs(testI, :);

        %[~, lateF_clsfOut{k,i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        lateF_clsfOut{enc,i} = hof_clsfOut{k, i} + hog_clsfOut{k, i} + mbhx_clsfOut{k, i} + mbhy_clsfOut{k, i};
        
        lateF_accuracy{enc,i} = ClassificationAccuracy(lateF_clsfOut{enc,i}, testLabs);
        fprintf('accuracy(%d,%d): %.3f\n', k,i, mean(lateF_accuracy{enc,i}));

        %all_clfsOut{k,i}=clfsOut;
        %all_accuracy{k,i}=accuracy;
    end
    
    enc=enc+1;
end

for k=1:3
    k
    for i=1:3
        trainI=splits(:,i)==1;
        testI=~trainI;

        %trainDist = allDist{k,i}(trainI, trainI);
        %testDist = allDist{k,i}(testI, trainI);
        trainLabs = labs(trainI,:);
        testLabs = labs(testI, :);

        %[~, lateF_clsfOut{k,i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        lateF_clsfOut{enc,i} = spVGG19_clsfOut{k, i} + tempVGG16_clsfOut{k, i};
        
        lateF_accuracy{enc,i} = ClassificationAccuracy(lateF_clsfOut{enc,i}, testLabs);
        fprintf('accuracy(%d,%d): %.3f\n', k,i, mean(lateF_accuracy{enc,i}));

        %all_clfsOut{k,i}=clfsOut;
        %all_accuracy{k,i}=accuracy;
    end
    
    enc=enc+1;
end


for k=1:3
    k
    for i=1:3
        trainI=splits(:,i)==1;
        testI=~trainI;

        %trainDist = allDist{k,i}(trainI, trainI);
        %testDist = allDist{k,i}(testI, trainI);
        trainLabs = labs(trainI,:);
        testLabs = labs(testI, :);

        %[~, lateF_clsfOut{k,i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        lateF_clsfOut{enc,i} = spVGG19_clsfOut{k, i} + tempVGG16_clsfOut{k, i} + hof_clsfOut{k, i} + hog_clsfOut{k, i} + mbhx_clsfOut{k, i} + mbhy_clsfOut{k, i};
        
        lateF_accuracy{enc,i} = ClassificationAccuracy(lateF_clsfOut{enc,i}, testLabs);
        fprintf('accuracy(%d,%d): %.3f\n', k,i, mean(lateF_accuracy{enc,i}));

        %all_clfsOut{k,i}=clfsOut;
        %all_accuracy{k,i}=accuracy;
    end
    
    enc=enc+1;
end


finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    mean_all_accuracy{j}=(lateF_accuracy{j,1} + lateF_accuracy{j,2} + lateF_accuracy{j,3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end

