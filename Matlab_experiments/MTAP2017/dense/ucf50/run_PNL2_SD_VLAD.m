

hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if ~isempty(findstr(hostname, 'cocoa'))
    rezPath='/home/ionut/asustor_ionut/Data/results/mtap2017/'
else if ~isempty(findstr(hostname, 'Halley'))
      rezPath='/home/ionut/asustor_ionut_2/Data/results/mtap2017/'
    end
end



% load([rezPath 'videoRep/FEVidHmgDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters_FV256numClusters_VLAD256_512_pcaDim72.mat']);
% hmg_vlad256=vlad256;
% hmg_vlad256f=vlad256f;
% hmg_vlad512=vlad512;
% hmg_vlad512f=vlad512f;
% hmg_b_f=b_f;
% hmg_sd=sd;
% hmg_sd_f=sd_f;



load([rezPath 'videoRep/FEVidHofDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters_FV256numClusters_VLAD256_512_pcaDim72.mat']);
hof_vlad256=vlad256;
hof_vlad256f=vlad256f;
hof_vlad512=vlad512;
hof_vlad512f=vlad512f;
hof_b_f=b_f;
hof_sd=sd;
hof_sd_f=sd_f;


load([rezPath 'videoRep/FEVidHogDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters_FV256numClusters_VLAD256_512_pcaDim72.mat']);
hog_vlad256=vlad256;
hog_vlad256f=vlad256f;
hog_vlad512=vlad512;
hog_vlad512f=vlad512f;
hog_b_f=b_f;
hog_sd=sd;
hog_sd_f=sd_f;

load([rezPath 'videoRep/FEVidMBHxDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters_FV256numClusters_VLAD256_512_pcaDim72.mat']);
mbhx_vlad256=vlad256;
mbhx_vlad256f=vlad256f;
mbhx_vlad512=vlad512;
mbhx_vlad512f=vlad512f;
mbhx_b_f=b_f;
mbhx_sd=sd;
mbhx_sd_f=sd_f;


load([rezPath 'videoRep/FEVidMBHyDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters_FV256numClusters_VLAD256_512_pcaDim72.mat']);
mbhy_vlad256=vlad256;
mbhy_vlad256f=vlad256f;
mbhy_vlad512=vlad512;
mbhy_vlad512f=vlad512f;
mbhy_b_f=b_f;
mbhy_sd=sd;
mbhy_sd_f=sd_f;

clear vlad256 vlad256f vlad512 vlad512f b_f sd sd_f fv 


alpha=0.1

descName={'hog', 'hof', 'mbhx', 'mbhy', 'hmg'}
encName={'_vlad256', '_vlad256f', '_vlad512', '_vlad512f', '_b_f', '_sd', '_sd_f'};

global DATAopts;
DATAopts = UCFInit;
addpath('./../');%!!!!!!!
[vids, labs, groups] = GetVideosPlusLabels('Full');

nEncoding=1;
allDist=cell(1, nEncoding);
all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


fileName=[rezPath  'results_VLADbazed_intraPNL2.txt']

parpool(5);

for dd=1:length(descName)
    for ee=1:length(encName)
        
        
        temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(eval(sprintf('%s%s', descName{dd},encName{ee})), 72), alpha));
        allDist{1}=temp * temp';
        
        
        for k=1:nEncoding

            % 
            % Leave-one-group-out cross-validation
            parfor i=1:max(groups)
                testI = groups == i;
                trainI = ~testI;
                trainDist = allDist{k}(trainI, trainI);
                testDist = allDist{k}(testI, trainI);
                trainLabs = labs(trainI,:);
                testLabs = labs(testI, :);

                [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
                accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
                fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
            end

            all_clfsOut{k}=clfsOut;
            all_accuracy{k}=accuracy;

            k
            perGroupAccuracy = mean(cat(2, accuracy{:}))'
            mean(mean(cat(2, accuracy{:})))

            fileID=fopen(fileName, 'a');
            fprintf(fileID, '%s%s: %.3f  ', ...
                         descName{dd},encName{ee}, mean(mean(cat(2, all_accuracy{k}{:}))) );
            fclose(fileID);

       end

            
        
        
    end
    fileID=fopen(fileName, 'a');
    fprintf(fileID, '\n' );
    fclose(fileID);
end

delete(gcp('nocreate'))
