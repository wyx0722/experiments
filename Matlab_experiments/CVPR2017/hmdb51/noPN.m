addpath('./../');%!!!!!!!


datasetName='HMDB51';


nEncoding=9;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(v256);
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(v319);
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(v512);
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(m256_abs);
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(m319_abs);
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(m512_abs);
allDist{6}=temp * temp';

temp=NormalizeRowsUnit(spV32_m);
allDist{7}=temp * temp';

temp=NormalizeRowsUnit(spV32);
allDist{8}=temp * temp';

temp=NormalizeRowsUnit(st_vlmpf32_abs);
allDist{9}=temp * temp';



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




%%%
for k=1:nEncoding
    k
    for i=1:3%parfor i=1:3 %\\\
        
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

%delete(gcp('nocreate')) %\\\\\\\
%%%%

finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end