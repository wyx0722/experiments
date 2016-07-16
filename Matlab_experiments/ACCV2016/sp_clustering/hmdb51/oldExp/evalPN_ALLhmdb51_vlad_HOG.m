
%% Do classification
nEncoding=7;
allDist=cell(1, nEncoding);
alpha

temp=NormalizeRowsUnit(PowerNormalization(v256, alpha));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v320, alpha));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(v512, alpha));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV8, alpha));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV32, alpha));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV64, alpha));
allDist{6}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(spV256, alpha));
allDist{7}=temp * temp';

clear temp

%each row for the cell represents the results for all 3 splits
all_clfsOut=cell(nEncoding,3);
all_accuracy=cell(nEncoding,3);
mean_all_clfsOut=cell(nEncoding,1);
mean_all_accuracy=cell(nEncoding,1);

cRange = 100;
nReps = 1;
nFolds = 3;


parpool(nEncoding);
parfor k=1:nEncoding
    k
    for i=1:3
        
        trainI = splits(:,i) == 1;
        testI  = splits(:,i) == 2;
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        trainDist = allDist{k}(trainI, trainI);
        testDist = allDist{k}(testI, trainI);
        

        [~, clfsOut] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        accuracy = ClassificationAccuracy(clfsOut, testLabs);
        fprintf('accuracy: %.3f\n', accuracy);

        all_clfsOut{k,i}=clfsOut;
        all_accuracy{k,i}=accuracy;
    end
end

delete(gcp('nocreate'))


finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    mean_all_clfsOut{j}=(all_clfsOut{j,1} + all_clfsOut{j,2} + all_clfsOut{j,3})./3;
    mean_all_accuracy{j}=(all_accuracy{j,1} + all_accuracy{j,2} + all_accuracy{j,3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end

