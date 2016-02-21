function [ all_clfsOut,  all_accuracy] = evalClassificationPNL2VLAD( file, alpha, pSize )

[vids, labs, groups] = GetVideosPlusLabels('Full');



pathF='/home/ionut/Data/results/CBMI2015_rezults/videoRep/';
VLADAll=load([pathF file]);

pathC='/home/ionut/Data/results/CBMI2015_rezults/clfsOut/';
load([pathC file]);
descParam

nEncoding=1;
parpool(pSize);

for p=1:length(alpha)
    
    p
    alpha(p)

n_VLADAll=NormalizeRowsUnit(PowerNormalization(VLADAll, alpha(p)));
allDist{1}=n_VLADAll * n_VLADAll';
clear n_VLADAll


all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


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

end





acc=mean(mean(cat(2, all_accuracy{1}{:})))

descType=func2str(descParam.Func);
try    
    
    fileName=['/home/ionut/experiments/Matlab_experiments/CBMI2016/results/classification_norm_PN/results_UCF50_norm__' 'Desc' descType '_norm' descParam.Normalisation 'VLAD.txt'];
    
    fileID=fopen(fileName, 'a');
    
    fprintf(fileID, '%s normBins:%s   PNL2 with alpha: %.2f --> acc: %.3f \r\n', ...
            descType, descParam.Normalisation, alpha(p), acc);
    
    fclose(fileID);
    
catch err
    
    fileName=['/home/ionut/experiments/Matlab_experiments/CBMI2016/results/classification_norm_PN/backup/results_UCF50_norm__' 'Desc' descType '_norm' descParam.Normalisation 'VLAD.txt'];
    
    fileID=fopen(fileName, 'a');
    
   fprintf(fileID, '%s normBins:%s   PNL2 with alpha: %.2f --> acc: %.3f \r\n', ...
            descType, descParam.Normalisation, alpha(p), acc);
    
    fclose(fileID);
    
    warning('error writing %s. Instead the file%s was saved in: ',err, fileName);
        
end

end

delete(gcp('nocreate'))





end

