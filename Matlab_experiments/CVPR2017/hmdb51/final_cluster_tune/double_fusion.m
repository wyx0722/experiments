
addpath('./../');%!!!!!!!
addpath('./../../');%!!!!!!!

global DATAopts;
DATAopts = HMDB51Init;
[allVids, labs, splits] = GetVideosPlusLabels();


bazeDir='/home/ionut/asustor_ionut/Data/results/cvpr2017/forFusion_state_of_the_art/hmdb51/clfsOut/'

name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerconv5bMediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetC3DpcaDim0spClusters2_4_8_16_32_64_128_256___clfsOut_sp32cl256PCA0.mat']
load(name);
c3d=clfsOut_sp32cl256PCA0(1); clear clfsOut_sp32cl256PCA0

name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetSpVGG19pcaDim0spClusters2_4_8_16_32_64_128_256___clfsOut_sp32cl256PCA0.mat']
load(name);
scn=clfsOut_sp32cl256PCA0(1); clear clfsOut_sp32cl256PCA0

name=[bazeDir 'FEVid_deepFeaturesClusters64_128_256_512_DatasetHMDB51Layerpool5MediaTypeDeepFNormFeatureMapsNoneNormalisationNonenetTempSplit1VGG16pcaDim0spClusters2_4_8_16_32_64_128_256___clfsOut_sp32cl256PCA0.mat']
load(name);
tcn=clfsOut_sp32cl256PCA0(1); clear clfsOut_sp32cl256PCA0



name=[bazeDir 'FEVidHmgDenseBlockSize8_8_6_DatasetHMDB51FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72SPfisherVector.mat']
load(name);
hmg=all_clfsOut(1); clear all_clfsOut

name=[bazeDir 'FEVid_IDTDatasetHMDB51IDTfeatureHOFMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim54fisherVector.mat']
load(name);
hof=all_clfsOut(1); clear all_clfsOut

name=[bazeDir 'FEVid_IDTDatasetHMDB51IDTfeatureHOGMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat']
load(name);
hog=all_clfsOut(1); clear all_clfsOut

name=[bazeDir 'FEVid_IDTDatasetHMDB51IDTfeatureMBHxMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat']
load(name);
mbhx=all_clfsOut(1); clear all_clfsOut

name=[bazeDir 'FEVid_IDTDatasetHMDB51IDTfeatureMBHyMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48fisherVector.mat']
load(name);
mbhy=all_clfsOut(1); clear all_clfsOut


name=[bazeDir 'all_early_fusion.mat']
load(name);
df=all_clfsOut(1);
df_hmg=all_clfsOut(4);
df_idt=all_clfsOut(5);
df_idt_hmg=all_clfsOut(6);



split1_trainI = splits(:,1) == 1; split1_testI  = splits(:,1) == 2;
split1_trainLabs = labs(split1_trainI,:); split1_testLabs = labs(split1_testI,:);

split2_trainI = splits(:,2) == 1; split2_testI  = splits(:,2) == 2;
split2_trainLabs = labs(split2_trainI,:); split2_testLabs = labs(split2_testI,:);

split3_trainI = splits(:,3) == 1; split3_testI  = splits(:,3) == 2;
split3_trainLabs = labs(split3_trainI,:); split3_testLabs = labs(split3_testI,:);


%df~~~~~~~~~~~~~~~~~~~~~~~~~~~~
split1_clfs=c3d{1}{1} + scn{1}{1} + tcn{1}{1};
split2_clfs=c3d{1}{2} + scn{1}{2} + tcn{1}{2};
split3_clfs=c3d{1}{3} + scn{1}{3} + tcn{1}{3};

sumLate_df=(mean(ClassificationAccuracy(split1_clfs, split1_testLabs)) + mean(ClassificationAccuracy(split2_clfs, split2_testLabs)) + ...
            mean(ClassificationAccuracy(split3_clfs, split3_testLabs)))/3; 

split1_clfs=c3d{1}{1} + scn{1}{1} + tcn{1}{1} + df{1}{1};
split2_clfs=c3d{1}{2} + scn{1}{2} + tcn{1}{2} + df{1}{2};
split3_clfs=c3d{1}{3} + scn{1}{3} + tcn{1}{3} + df{1}{3};

sumDouble_df=(mean(ClassificationAccuracy(split1_clfs, split1_testLabs)) + mean(ClassificationAccuracy(split2_clfs, split2_testLabs)) + ...
            mean(ClassificationAccuracy(split3_clfs, split3_testLabs)))/3;
%~~~~~~~~~~~~~~~~~~

%df+hmg~~~~~~~~~~~~
split1_clfs=c3d{1}{1} + scn{1}{1} + tcn{1}{1} + hmg{1}{1};
split2_clfs=c3d{1}{2} + scn{1}{2} + tcn{1}{2} + hmg{1}{2};
split3_clfs=c3d{1}{3} + scn{1}{3} + tcn{1}{3} + hmg{1}{3};

sumLate_df_hmg=(mean(ClassificationAccuracy(split1_clfs, split1_testLabs)) + mean(ClassificationAccuracy(split2_clfs, split2_testLabs)) + ...
            mean(ClassificationAccuracy(split3_clfs, split3_testLabs)))/3; 

split1_clfs=c3d{1}{1} + scn{1}{1} + tcn{1}{1} + hmg{1}{1} + df_hmg{1}{1};
split2_clfs=c3d{1}{2} + scn{1}{2} + tcn{1}{2} + hmg{1}{2} + df_hmg{1}{2};
split3_clfs=c3d{1}{3} + scn{1}{3} + tcn{1}{3} + hmg{1}{3} + df_hmg{1}{3};

sumDouble_df_hmg=(mean(ClassificationAccuracy(split1_clfs, split1_testLabs)) + mean(ClassificationAccuracy(split2_clfs, split2_testLabs)) + ...
            mean(ClassificationAccuracy(split3_clfs, split3_testLabs)))/3;
%~~~~~~~~~~~~~~~~~~

%df+idt~~~~~~~~~~~~
split1_clfs=c3d{1}{1} + scn{1}{1} + tcn{1}{1} + hof{1}{1} + hog{1}{1} + mbhx{1}{1} + mbhy{1}{1};
split2_clfs=c3d{1}{2} + scn{1}{2} + tcn{1}{2} + hof{1}{2} + hog{1}{2} + mbhx{1}{2} + mbhy{1}{2};
split3_clfs=c3d{1}{3} + scn{1}{3} + tcn{1}{3} + hof{1}{3} + hog{1}{3} + mbhx{1}{3} + mbhy{1}{3};

sumLate_df_idt=(mean(ClassificationAccuracy(split1_clfs, split1_testLabs)) + mean(ClassificationAccuracy(split2_clfs, split2_testLabs)) + ...
            mean(ClassificationAccuracy(split3_clfs, split3_testLabs)))/3; 

split1_clfs=c3d{1}{1} + scn{1}{1} + tcn{1}{1} + hof{1}{1} + hog{1}{1} + mbhx{1}{1} + mbhy{1}{1} + df_idt{1}{1};
split2_clfs=c3d{1}{2} + scn{1}{2} + tcn{1}{2} + hof{1}{2} + hog{1}{2} + mbhx{1}{2} + mbhy{1}{2} + df_idt{1}{2};
split3_clfs=c3d{1}{3} + scn{1}{3} + tcn{1}{3} + hof{1}{3} + hog{1}{3} + mbhx{1}{3} + mbhy{1}{3} + df_idt{1}{3};

sumDouble_df_idt=(mean(ClassificationAccuracy(split1_clfs, split1_testLabs)) + mean(ClassificationAccuracy(split2_clfs, split2_testLabs)) + ...
            mean(ClassificationAccuracy(split3_clfs, split3_testLabs)))/3;
%~~~~~~~~~~~~~~~~~~

%df+idt+hmg~~~~~~~~~~~~
split1_clfs=c3d{1}{1} + scn{1}{1} + tcn{1}{1} + hof{1}{1} + hog{1}{1} + mbhx{1}{1} + mbhy{1}{1} + hmg{1}{1};
split2_clfs=c3d{1}{2} + scn{1}{2} + tcn{1}{2} + hof{1}{2} + hog{1}{2} + mbhx{1}{2} + mbhy{1}{2} + hmg{1}{2};
split3_clfs=c3d{1}{3} + scn{1}{3} + tcn{1}{3} + hof{1}{3} + hog{1}{3} + mbhx{1}{3} + mbhy{1}{3} + hmg{1}{3};

sumLate_df_idt_hmg=(mean(ClassificationAccuracy(split1_clfs, split1_testLabs)) + mean(ClassificationAccuracy(split2_clfs, split2_testLabs)) + ...
            mean(ClassificationAccuracy(split3_clfs, split3_testLabs)))/3; 

split1_clfs=c3d{1}{1} + scn{1}{1} + tcn{1}{1} + hof{1}{1} + hog{1}{1} + mbhx{1}{1} + mbhy{1}{1} + hmg{1}{1} + df_idt_hmg{1}{1};
split2_clfs=c3d{1}{2} + scn{1}{2} + tcn{1}{2} + hof{1}{2} + hog{1}{2} + mbhx{1}{2} + mbhy{1}{2} + hmg{1}{2} + df_idt_hmg{1}{2};
split3_clfs=c3d{1}{3} + scn{1}{3} + tcn{1}{3} + hof{1}{3} + hog{1}{3} + mbhx{1}{3} + mbhy{1}{3} + hmg{1}{3} + df_idt_hmg{1}{3};

sumDouble_df_idt_hmg=(mean(ClassificationAccuracy(split1_clfs, split1_testLabs)) + mean(ClassificationAccuracy(split2_clfs, split2_testLabs)) + ...
            mean(ClassificationAccuracy(split3_clfs, split3_testLabs)))/3;
%~~~~~~~~~~~~~~~~~~


interval=0:0.1:1;


%~~~~~~Deep features: SCN+TCN+C3D

L_df_weights=getSolutions_wFusion(interval,3);

