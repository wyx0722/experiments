

hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if ~isempty(findstr(hostname, 'cocoa'))
    rezPath='/home/ionut/asustor_ionut/Data/results/mtap2017/'
else if ~isempty(findstr(hostname, 'Halley'))
      rezPath='/home/ionut/asustor_ionut_2/Data/results/mtap2017/'
    end
end




alpha=0.5

descName={'Hog', 'Hof', 'MBHx', 'MBHy', 'Hmg'}
fsr=[2 3 6];


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


fileName=[rezPath  'FSR_SD_VLAD_PNL05.txt']

parpool(5);

for dd=1:length(descName)
    fileID=fopen(fileName, 'a');
    fprintf(fileID, '%s:\n ', descName{dd});
    fclose(fileID);
    for f=1:length(fsr)
        
        
        nameLoad=[rezPath 'videoRep/VLADbased/' 'FEVid' descName{dd} 'DenseBlockSize8_8_' 6/fsr(f) '_FrameSampleRate' fsr(f) 'MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72_SD_VLAD__SD_VLADAll.mat']
        load(nameLoad);
        
        temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(SD_VLADAll, 72), alpha));
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
            fprintf(fileID, '(%g %g): %.3f  ', ...
                         6/fsr(f),fsr(f), mean(mean(cat(2, all_accuracy{k}{:}))) );
            fclose(fileID);

       end

            
        
        
    end
    fileID=fopen(fileName, 'a');
    fprintf(fileID, '\n\n' );
    fclose(fileID);
end

delete(gcp('nocreate'))
