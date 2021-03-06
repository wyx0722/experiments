alpha=0.1:0.1:1

nEncoding=2*length(alpha);

allDist=cell(1, nEncoding);

for i=1:length(alpha)
    i
    temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(tempVLAD1, alpha(i))), NormalizeRowsUnit(PowerNormalization(spVGG19VLAD1, alpha(i)))) );
    allDist{2*i-1}=temp * temp';

    temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit(PowerNormalization(tempVLAD2, alpha(i))), NormalizeRowsUnit(PowerNormalization(spVGG19VLAD2, alpha(i)))) );
    allDist{2*i}=temp * temp';
    
end



all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;

parpool(nEncoding);
parfor k=1:nEncoding
    k
    trainI=trainTestSplit==1;
    testI=~trainI;
    
    trainDist = allDist{k}(trainI, trainI);
    testDist = allDist{k}(testI, trainI);
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy = ClassificationAccuracy(clfsOut, testLabs);
    fprintf('accuracy: %.3f\n', accuracy);
    
    all_clfsOut{k}=clfsOut;
    all_accuracy{k}=accuracy;
    
    k
    mean(accuracy)
end

delete(gcp('nocreate'))



fileName=sprintf('/home/ionut/experiments/Matlab_experiments/ICPR2016/results/results_UCF101_Split1_evalPNalpha_early_fusion_TempVGG16Split1_SpVGG19_PCAdim256_Clusters256_normROOTSIFT_DA_VLAD.txt'); 
fileID=fopen(fileName, 'a');
    

for i=1:length(alpha)
    acc1=mean(all_accuracy{2*i-1});
    acc2=mean(all_accuracy{2*i});
    fprintf(fileID, 'early fusion: TempVGG16Split1 + SpVGG19  PowerNormalization with alpha=%.2f ---->   acc VLAD: %.4f    acc DA-VLAD(1): %.4f \r\n', alpha(i), acc1, acc2);   
end
fclose(fileID);


