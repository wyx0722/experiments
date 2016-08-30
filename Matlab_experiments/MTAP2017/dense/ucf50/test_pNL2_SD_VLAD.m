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


descPath=[rezPath 'videoRep/VLADbased/encVideoReps/']


descFile{1}='FEVidhogDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8pcaDim72numClusters_256_sd.mat';
descFile{2}='FEVidhofDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8pcaDim72numClusters_256_sd.mat';
descFile{3}='FEVidmbhxDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8pcaDim72numClusters_256_sd.mat';
descFile{4}='FEVidmbhyDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8pcaDim72numClusters_256_sd.mat';
descFile{5}='FEVidhmgDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8pcaDim72numClusters_256_sd.mat';

descName={'hog', 'hof', 'mbhx', 'mbhy', 'hmg'}
descVal=cell(1, length(descName));

alpha=[0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1];
nEncoding=9;
allDist=cell(1, nEncoding);


fileName=[rezPath  'results_PNL2_SD-VLADintra.txt']

parpool(5);

%% Do classification


for j=1:length(descFile)
    j
    
    load([descPath descFile{j}]);
    descVal{j}=encVideoRep;
    
    for i=1:length(alpha)
        i
        
        temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(encVideoRep, 72), alpha(i)));
    

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
