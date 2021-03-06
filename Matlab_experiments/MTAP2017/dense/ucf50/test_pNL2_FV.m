global DATAopts;
DATAopts = UCFInit;
addpath('./../');%!!!!!!!
[vids, labs, groups] = GetVideosPlusLabels('Full');



hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if ~isempty(findstr(hostname, 'cocoa'))
    rezPath='/home/ionut/asustor_ionut/Data/results/mtap2017/'
else if ~isempty(findstr(hostname, 'Halley'))
      rezPath='/home/ionut/asustor_ionut_2/Data/results/mtap2017/'
    end
end


descPath=[rezPath 'videoRep/']


descFile{1}='FEVidHmgDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72.mat';
descFile{2}='FEVidHofDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72.mat';
descFile{3}='FEVidHogDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72.mat';
descFile{4}='FEVidMBHxDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72.mat';
descFile{5}='FEVidMBHyDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72.mat';

alpha=[0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1];
nEncoding=9;
allDist=cell(1, nEncoding);


fileName=[rezPath  'results_PNL2_FV_intraPNL2.txt']

parpool(5);

%% Do classification


for j=1:length(descFile)

    load([descPath descFile{j}]);

    for i=1:length(alpha)
        i

        temp=NormalizeRowsUnit(PowerNormalization(fisherAll, alpha(i)));


        allDist{i}=temp * temp';

    end


    all_clfsOut=cell(1,nEncoding);
    all_accuracy=cell(1,nEncoding);

    cRange = 100;
    nReps = 1;
    nFolds = 3;




    fileID=fopen(fileName, 'a');
    fprintf(fileID, '%s \n PNL2 norm before classification (alpha): \n', descFile{j});
    fclose(fileID);






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
    fprintf(fileID, '%.2f--> %.3f  ', ...
                 alpha(k), mean(mean(cat(2, all_accuracy{k}{:}))) );
    fclose(fileID);

    end

    fileID=fopen(fileName, 'a');
    fprintf(fileID, '\n\n' );
    fclose(fileID);


end
delete(gcp('nocreate'))
