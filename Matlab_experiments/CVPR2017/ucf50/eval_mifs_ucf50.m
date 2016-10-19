% 
% path_features='/home/ionut/asustor_ionut/Data/MIFS/ucf50_l3/'
% 
% dirFeatures=sprintf('%scache/*.mat',path_features);
% 
% list_mat_file=dir(dirFeatures);
% 
% cell_features=cell(1, length(list_mat_file));
% 
% fprintf('Load features from %d files!!  ... \n', length(list_mat_file));
% for i=1:length(list_mat_file)
%      fprintf('%d ', i);
%      filename=[path_features list_mat_file(i).name];
%      load(filename);
%      cell_features{i}=data';    
% end
% 
% 
% all_data=cat(1, cell_features{:});


% load([ path_features 'imdb.mat']);
% 
% 
% 
% global DATAopts;
% DATAopts = UCFInit;
% 
% 
% [allVids, labs, groups] = GetVideosPlusLabels('Full');
% 
% representation=zeros(length(allVids), size(all_data, 2));
% 
% for i=1:length(allVids)
%     if mod(i, 100)==0
%         fprintf('%d ', i);
%     end
%     
%     for j=1:(length(images.name))
%         
%         if ~isempty(strfind([meta.classes{images.class(j)} '/' images.name{j}(1:end-4)], allVids{i}))
%             representation(i, :)=all_data(j, :);
%             continue
%         end
%         
%     end 
%     
%     
% end

nPar=5;

nEncoding=2;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(representation);
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(representation, alpha));
allDist{2}=temp * temp';

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;

parpool(nPar);
for k=1:nEncoding
k
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

fprintf('Accuracy for encoding %d: %.3f\n',k, mean(mean(cat(2, accuracy{:}))'));

end

delete(gcp('nocreate'))

clear allDist



finalAcc=zeros(1,nEncoding);
for j=1:nEncoding

    finalAcc(j)=mean(mean(cat(2, all_accuracy{j}{:}), 2));
    fprintf('%.3f\n', finalAcc(j));

    
end

 
 
 
