hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if ~isempty(findstr(hostname, 'cocoa'))
    rezPath='/home/ionut/asustor_ionut/Data/results/mtap2017/ucf50/'
else if ~isempty(findstr(hostname, 'Halley'))
      rezPath='/home/ionut/asustor_ionut_2/Data/results/mtap2017/ucf50/'
    end
end


encName={ '_sd_f', '_b_f'};
descName={'hmg', 'hog', 'hof', 'mbhx', 'mbhy'}


global DATAopts;
DATAopts = UCFInit;
addpath('./../');%!!!!!!!
[vids, labs, groups] = GetVideosPlusLabels('Full');

nEncoding=5;
allDist=cell(1, nEncoding);
all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;

alpha=0.5;

fileName=[rezPath  'evalParts.txt']



for dd=1:length(descName)
    
    
    %%%%%%%%%%%
    fileName_sd_f=[rezPath 'videoRep/VLADbased/encVideoReps/' 'FEVid' descName{dd} 'DenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8pcaDim72numClusters_256' '_sd_f.mat']
    load(fileName_sd_f);
    
    sizePart=size(encVideoRep,2)/4;
    sp1=encVideoRep(:,1:sizePart);
    sp2=encVideoRep(:,sizePart+1:2*sizePart);
    sp3=encVideoRep(:,2*sizePart+1:3*sizePart);
    sp4=encVideoRep(:,3*sizePart:end);
    
    size(sp1)
    isequal(size(sp1),size(sp2), size(sp3), size(sp4))
    
    sd1=cat(2, sp1(:, 1:size(sp1,2)/2), sp2(:, 1:size(sp2,2)/2), sp3(:, 1:size(sp3,2)/2), sp4(:, 1:size(sp4,2)/2));
    sd2=cat(2, sp1(:,(size(sp1,2)/2)+1:end), sp2(:,(size(sp2,2)/2)+1:end), sp3(:,(size(sp3,2)/2)+1:end), sp4(:,(size(sp4,2)/2)+1:end));
    
    temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(sd1, 72), alpha));
    allDist{1}=temp * temp';
    
    temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(sd2, 72), alpha));
    allDist{2}=temp * temp';
    %%%%%%%%%%%%%%
    
    %%%%%%%%%%%
    fileName_b_f=[rezPath 'videoRep/VLADbased/encVideoReps/' 'FEVid' descName{dd} 'DenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8pcaDim72numClusters_256' '_b_f.mat']
    load(fileName_b_f);
    
    sizePart=size(encVideoRep,2)/4;
    sp1=encVideoRep(:,1:sizePart);
    sp2=encVideoRep(:,sizePart+1:2*sizePart);
    sp3=encVideoRep(:,2*sizePart+1:3*sizePart);
    sp4=encVideoRep(:,3*sizePart:end);
    
    size(sp1)
    isequal(size(sp1),size(sp2), size(sp3), size(sp4))
    
    le=size(sp1,2)/3;
    bf1=cat(2, sp1(:, 1:le), sp2(:, 1:le), sp3(:, 1:le), sp4(:, 1:le));
    bf2=cat(2, sp1(:,le+1:2*le), sp2(:,le+1:2*le), sp3(:,le+1:2*le), sp4(:,le+1:2*le));
    bf3=cat(2, sp1(:,2*le+1:end), sp2(:,2*le+1:end), sp3(:,2*le+1:end), sp4(:,2*le+1:end));
    
    temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(bf1, 72), alpha));
    allDist{3}=temp * temp';
    
    temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(bf2, 72), alpha));
    allDist{4}=temp * temp';
    
    temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(bf3, 72), alpha));
    allDist{5}=temp * temp';
    %%%%%%%%%%%%%%
    parpool(5);
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



    end
    delete(gcp('nocreate'))
    

    fileID=fopen(fileName, 'a');
    fprintf(fileID, '%s: \n SD-VLAD1:%.3f  SD-VLAD2:%.3f  H-VLAD1:%.3f  H-VLAD2:%.3f  H-VLAD3:%.3f  \n\n', ...
                 descName{dd}, mean(mean(cat(2, all_accuracy{1}{:}))), mean(mean(cat(2, all_accuracy{2}{:}))), ...
                 mean(mean(cat(2, all_accuracy{3}{:}))), mean(mean(cat(2, all_accuracy{4}{:}))), mean(mean(cat(2, all_accuracy{5}{:}))) );
    fclose(fileID);        
        
   

end

