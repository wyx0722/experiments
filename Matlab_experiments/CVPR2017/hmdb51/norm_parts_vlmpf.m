

norm_st_vlmpf32_noPCA_c3d=zeros(size(st_vlmpf32_noPCA_c3d));
norm_st_vlmpf32_noPCA_c3d(:, 1:(256*512))=NormalizeRowsUnit(st_vlmpf32_noPCA_c3d(:, 1:(256*512)));
norm_st_vlmpf32_noPCA_c3d(:, ((256*512)+1):(256*512)+(32*512))=NormalizeRowsUnit(st_vlmpf32_noPCA_c3d(:, ((256*512)+1):(256*512)+(32*512)) );
norm_st_vlmpf32_noPCA_c3d(:,(256*512)+(32*512)+1 :(256*512)+(32*512)+(32*256))=NormalizeRowsUnit(st_vlmpf32_noPCA_c3d(:,(256*512)+(32*512)+1 :(256*512)+(32*512)+(32*256)) );


norm_st_vlmpf32_noPCA_sp=zeros(size(st_vlmpf32_noPCA_sp));
norm_st_vlmpf32_noPCA_sp(:, 1:(256*512))=NormalizeRowsUnit(st_vlmpf32_noPCA_sp(:, 1:(256*512)));
norm_st_vlmpf32_noPCA_sp(:, ((256*512)+1):(256*512)+(32*512))=NormalizeRowsUnit(st_vlmpf32_noPCA_sp(:, ((256*512)+1):(256*512)+(32*512)) );
norm_st_vlmpf32_noPCA_sp(:,(256*512)+(32*512)+1 :(256*512)+(32*512)+(32*256))=NormalizeRowsUnit(st_vlmpf32_noPCA_sp(:,(256*512)+(32*512)+1 :(256*512)+(32*512)+(32*256)) );

norm_st_vlmpf32_noPCA_temp=zeros(size(st_vlmpf32_noPCA_temp));
norm_st_vlmpf32_noPCA_temp(:, 1:(256*512))=NormalizeRowsUnit(st_vlmpf32_noPCA_temp(:, 1:(256*512)));
norm_st_vlmpf32_noPCA_temp(:, ((256*512)+1):(256*512)+(32*512))=NormalizeRowsUnit(st_vlmpf32_noPCA_temp(:, ((256*512)+1):(256*512)+(32*512)) );
norm_st_vlmpf32_noPCA_temp(:,(256*512)+(32*512)+1 :(256*512)+(32*512)+(32*256))=NormalizeRowsUnit(st_vlmpf32_noPCA_temp(:,(256*512)+(32*512)+1 :(256*512)+(32*512)+(32*256)) );


nEncoding=4;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit( norm_st_vlmpf32_noPCA_c3d);
allDist{1}=temp * temp';

temp=NormalizeRowsUnit( norm_st_vlmpf32_noPCA_sp);
allDist{2}=temp * temp';

temp=NormalizeRowsUnit( norm_st_vlmpf32_noPCA_temp);
allDist{3}=temp * temp';

temp=NormalizeRowsUnit( cat(2, NormalizeRowsUnit( norm_st_vlmpf32_noPCA_temp), NormalizeRowsUnit( norm_st_vlmpf32_noPCA_sp),...
    NormalizeRowsUnit( norm_st_vlmpf32_noPCA_c3d) ) );
allDist{4}=temp * temp';



clear temp

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



%%%
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

delete(gcp('nocreate'))
%%%%

finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('Encoding %d --> MAcc: %.3f \n', j, finalAcc(j));
end






