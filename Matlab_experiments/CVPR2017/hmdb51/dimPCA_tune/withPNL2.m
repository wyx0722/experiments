
    
alpha=0.5

nEncoding=8;
allDist=cell(1, nEncoding);



temp=NormalizeRowsUnit(PowerNormalization(sp32cl256pca64(:, 1:size(cell_Clusters{1}.vocabulary, 1) * size(cell_Clusters{1}.vocabulary, 2) ), alpha));
allDist{1}=temp * temp';
temp=NormalizeRowsUnit(PowerNormalization(sp32cl256pca128(:, 1:size(cell_Clusters{2}.vocabulary, 1) * size(cell_Clusters{2}.vocabulary, 2) ), alpha));
allDist{2}=temp * temp';
temp=NormalizeRowsUnit(PowerNormalization(sp32cl256pca256(:, 1:size(cell_Clusters{3}.vocabulary, 1) * size(cell_Clusters{3}.vocabulary, 2) ), alpha));
allDist{3}=temp * temp';
temp=NormalizeRowsUnit(PowerNormalization(sp32cl256pca0(:, 1:size(cell_Clusters{4}.vocabulary, 1) * size(cell_Clusters{4}.vocabulary, 2) ), alpha)); 
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(sp32cl256pca64, alpha)); allDist{5}=temp * temp';
temp=NormalizeRowsUnit(PowerNormalization(sp32cl256pca128, alpha)); allDist{6}=temp * temp';
temp=NormalizeRowsUnit(PowerNormalization(sp32cl256pca256, alpha)); allDist{7}=temp * temp';
temp=NormalizeRowsUnit(PowerNormalization(sp32cl256pca0, alpha)); allDist{8}=temp * temp';



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



parpool(3);
%%
for k=1:nEncoding
    k
    parfor i=1:3
        
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

delete(gcp('nocreate')) %///
%%%%

finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('%.3f\n', finalAcc(j));

    
end



 bazeSavePath='/home/ionut/asustor_ionut/Data/results/cvpr2017/hmdb51/dimPCA/';
 
 clfsOut_sp32cl256pca0 = all_clfsOut(8);
 acc_sp32cl256pca0 = all_accuracy(8);
 
 intructions_compute_acc='mean((acc_sp32cl256pca0{1}{1} + acc_sp32cl256pca0{1}{2} + acc_sp32cl256pca0{1}{3})./3)';
 
saveName = [bazeSavePath 'clfsOut/'  DescParam2Name(descParam) '_PNL2__sp32cl256pca0.mat']
save(saveName, '-v7.3', 'descParam', 'clfsOut_sp32cl256pca0', 'acc_sp32cl256pca0', 'intructions_compute_acc');

saveName2 = [bazeSavePath 'videoRep/'  DescParam2Name(descParam) '__sp32cl256pca0.mat']
save(saveName2, '-v7.3', 'descParam', 'sp32cl256pca0');


