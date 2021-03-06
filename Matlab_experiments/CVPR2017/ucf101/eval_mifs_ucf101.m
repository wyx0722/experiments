
% alpha=0.1;
% global DATAopts;
% DATAopts = UCF101Init;
% 
% 
% datasetName='UCF101';
% 
% 
% [allVids, labs, splits] = GetVideosPlusLabels('Challenge');
% 


path_features='/home/ionut/asustor_ionut/Data/MIFS/ucf101_l3/'




dirFeatures=sprintf('%scache/*.mat',path_features);

list_mat_file=dir(dirFeatures);

cell_features=cell(1, length(list_mat_file));

fprintf('Load features from %d files!!  ... \n', length(list_mat_file));
for i=1:length(list_mat_file)
     fprintf('%d ', i);
     filename=[path_features 'cache/' list_mat_file(i).name];
     load(filename);
     cell_features{i}=data';    
end


all_data=cat(1, cell_features{:});


load([ path_features 'imdb.mat']);

representation=zeros(length(allVids), size(all_data, 2));

for i=1:length(allVids)
    if mod(i, 100)==0
        fprintf('%d ', i);
    end
    
    for j=1:(length(images.name))
        
        if ~isempty(strfind([meta.classes{images.class(j)} '/' images.name{j}(1:end-4)], allVids{i}))
            representation(i, :)=all_data(j, :);
            continue
        end
        
    end 
    
    
end




alpha=0.1;
n_representation=NormalizeRowsUnit(PowerNormalization(representation, alpha));


nEncoding=10;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(representation);
allDist{7,1}=temp * temp'; allDist{7,2}=allDist{7,1};  allDist{7,3}=allDist{7,1};

temp=n_representation;
allDist{8,1}=temp * temp'; allDist{8,2}=allDist{8,1};  allDist{8,3}=allDist{8,1};


temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_representation ));
allDist{9,1}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit2VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_representation  ));
allDist{9,2}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit3VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_representation  ));
allDist{9,3}=temp * temp';

temp=NormalizeRowsUnit( cat(2, n_TempSplit1VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hmg_fv, n_representation ));
allDist{10,1}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit2VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hmg_fv, n_representation  ));
allDist{10,2}=temp * temp';
temp=NormalizeRowsUnit( cat(2, n_TempSplit3VGG16_sp32cl256pca0, n_SpVGG19_sp32cl256pca0, n_C3D_sp32cl256pca0, n_hmg_fv, n_representation  ));
allDist{10,3}=temp * temp';






% %each row for the cell represents the results for all 3 splits
% all_clfsOut=cell(1,nEncoding);
% all_accuracy=cell(1,nEncoding);
% clfsOut=cell(1,nEncoding);
% accuracy=cell(1,nEncoding);
% %mean_all_clfsOut=cell(nEncoding,1);
% mean_all_accuracy=cell(nEncoding,1);

cRange = 100;
nReps = 1;
nFolds = 3;



parpool(3);
%%
for k=7:nEncoding %!!!!!!!!!
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
        
        trainDist = allDist{k,i}(trainI, trainI);
        testDist = allDist{k,i}(testI, trainI);
        

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