

nEncoding=11;
allDist=cell(1, nEncoding);


t_feature=sp2cl256;
t_feature(:, end-(size(cell_spClusters{1}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{1}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{1}=temp * temp';

t_feature=sp4cl256;
t_feature(:, end-(size(cell_spClusters{2}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{2}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{2}=temp * temp';

t_feature=sp8cl256;
t_feature(:, end-(size(cell_spClusters{3}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{3}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{3}=temp * temp';

t_feature=sp16cl256;
t_feature(:, end-(size(cell_spClusters{4}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{4}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{4}=temp * temp';

t_feature=sp32cl256;
t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{5}=temp * temp';

t_feature=sp64cl256;
t_feature(:, end-(size(cell_spClusters{6}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{6}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{6}=temp * temp';

t_feature=sp128cl256;
t_feature(:, end-(size(cell_spClusters{7}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{7}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{7}=temp * temp';

t_feature=sp256cl256;
t_feature(:, end-(size(cell_spClusters{8}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{8}.vocabulary, 1)*size(cell_Clusters{3}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{8}=temp * temp';

    
t_feature=sp32cl64;
t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{1}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{1}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{9}=temp * temp';

t_feature=sp32cl128;
t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{2}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{2}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{10}=temp * temp';

t_feature=sp32cl512;
t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{4}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{5}.vocabulary, 1)*size(cell_Clusters{4}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{11}=temp * temp';

clear t_feature

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
     fprintf('Accuracy for encoding %d: %.3f\n',k, mean((all_accuracy{k}{1} + all_accuracy{k}{2} + all_accuracy{k}{3})./3));
end

delete(gcp('nocreate')) %///
%%%%

clear allDist


finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('%.3f\n', finalAcc(j));

    
end

