
addpath('./../');%!!!!!!!

global DATAopts;

datasetName='HMDB51'

if ~isempty(strfind(datasetName, 'HMDB51'))
    DATAopts = HMDB51Init;
    [allVids, labs, splits] = GetVideosPlusLabels();
elseif ~isempty(strfind(datasetName, 'UCF101'))
    DATAopts = UCF101Init;
    [allVids, labs, splits] = GetVideosPlusLabels('Challenge');
end



load_name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/dense/FEVidHogDenseBlockSize8_8_1_DatasetHMDB51FrameSampleRate6MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72SPfisherVector.mat'
load(load_name);
hog_fv_6=fisherVector;
clear fisherVector

load_name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/dense/FEVidHmgDenseBlockSize8_8_6_DatasetHMDB51FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72SPfisherVector.mat'
load(load_name);
hmg_fv_1=fisherVector;
clear fisherVector


load_name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/dense/FEVidHogDenseBlockSize8_8_1_DatasetHMDB51FrameSampleRate6MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72SD_VLAD.mat'
load(load_name);
hog_sd_vlad_6=SD_VLADAll;
clear SD_VLADAll

load_name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/dense/FEVidHmgDenseBlockSize8_8_1_DatasetHMDB51FrameSampleRate6MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72SD_VLAD.mat'
load(load_name);
hmg_sd_vlad_6=SD_VLADAll;
clear SD_VLADAll


load_name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/dense/FEVidHofDenseBlockSize8_8_2_DatasetHMDB51FrameSampleRate3MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72SD_VLAD.mat'
load(load_name);
hof_sd_vlad_3=SD_VLADAll;
clear SD_VLADAll

load_name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/dense/FEVidMBHxDenseBlockSize8_8_2_DatasetHMDB51FrameSampleRate3MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72SD_VLAD.mat'
load(load_name);
mbhx_sd_vlad_3=SD_VLADAll;
clear SD_VLADAll

load_name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/dense/FEVidMBHyDenseBlockSize8_8_2_DatasetHMDB51FrameSampleRate3MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72SD_VLAD.mat'
load(load_name);
mbhy_sd_vlad_3=SD_VLADAll;
clear SD_VLADAll


load_name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/iDT/FEVid_IDTDatasetHMDB51IDTfeatureHOFMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim54fisherVector.mat'
load(load_name);
hof_iDT_fv=fisherVector;
clear fisherVector

load_name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/iDT/FEVid_IDTDatasetHMDB51IDTfeatureHOGMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat'
load(load_name);
hog_iDT_fv=fisherVector;
clear fisherVector

load_name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/iDT/FEVid_IDTDatasetHMDB51IDTfeatureMBHxMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat'
load(load_name);
mbhx_iDT_fv=fisherVector;
clear fisherVector

load_name='/home/ionut/asustor_ionut/Data/results/mtap2017/hmdb51/videoRep/iDT/FEVid_IDTDatasetHMDB51IDTfeatureMBHyMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat'
load(load_name);
mbhy_iDT_fv=fisherVector;
clear fisherVector



%% Do classification

nEncoding=5;
allDist=cell(1, nEncoding);


temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(hog_fv_6, 0.1)), NormalizeRowsUnit(PowerNormalization(hmg_fv_1, 0.1))  ));
allDist{1}=temp * temp';


temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(hog_sd_vlad_6, 72), 0.5)), ...
       NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(hmg_sd_vlad_6, 72), 0.5)), ...
       NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(hof_sd_vlad_3, 72), 0.5)), ...
       NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(mbhx_sd_vlad_3, 72), 0.5)), ...
       NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(mbhy_sd_vlad_3, 72), 0.5)) ...
       ));
allDist{2}=temp * temp';


temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(hog_iDT_fv, 0.1)), NormalizeRowsUnit(PowerNormalization(hof_iDT_fv, 0.1)), ...
    NormalizeRowsUnit(PowerNormalization(mbhx_iDT_fv, 0.1)), NormalizeRowsUnit(PowerNormalization(mbhy_iDT_fv, 0.1))));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(hog_iDT_fv, 0.5)), NormalizeRowsUnit(PowerNormalization(hof_iDT_fv, 0.5)), ...
    NormalizeRowsUnit(PowerNormalization(mbhx_iDT_fv, 0.5)), NormalizeRowsUnit(PowerNormalization(mbhy_iDT_fv, 0.5))));
allDist{4}=temp * temp';



temp=NormalizeRowsUnit(cat(2, NormalizeRowsUnit(PowerNormalization(hog_iDT_fv, 0.1)), NormalizeRowsUnit(PowerNormalization(hof_iDT_fv, 0.1)), ...
    NormalizeRowsUnit(PowerNormalization(mbhx_iDT_fv, 0.1)), NormalizeRowsUnit(PowerNormalization(mbhy_iDT_fv, 0.1)), ...
    NormalizeRowsUnit(PowerNormalization(hmg_fv_1, 0.1)) ));
allDist{5}=temp * temp';


%each row for the cell represents the results for all 3 splits
all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);
clfsOut=cell(1,nEncoding);
accuracy=cell(1,nEncoding);
%mean_all_clfsOut=cell(nEncoding,1);
mean_all_accuracy=cell(nEncoding,1);

cRange = 100;
nReps = 1;
nFolds = 3;



parpool(3);
%%%
for k=1:nEncoding
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



finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end
