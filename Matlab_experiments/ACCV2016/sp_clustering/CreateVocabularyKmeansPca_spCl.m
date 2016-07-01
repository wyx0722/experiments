function [cell_Clusters, cell_spClusters, pcaMap] = CreateVocabularyKmeansPca_spCl(imNames, descParam, ...
                                                         numDescriptors, forceCreate)
% [vocabulary, pcaMap] = CreateVocabularyKmeansPca(imNames, descParam, numClusters, pcaDim, ...
%                                                          numDescriptors, forceCreate)
%
% Create Kmeans vocabulary for use in VLAD encoding, hard-, or soft-
% assignment
%
% imNames:              Image names from which to learn Kmeans clustering
% descParam:            Structure defining type of local descriptors
% numClusters:          Number of clusters for Kmeans clusters
% pcaDim:               PCA reduction on descriptors.
% numDescriptors (optional): Number of descriptors to learn Kmeans from. Default: 10^6
% forceCreate(optional):Forces Kmeans to be recreated even if already exists
%
% vocabulary:           Kmeans vocabulary
% pcaMap:               Corresponding PCA mapping
%

global DATAopts;

vocabularySaveName = sprintf('%sKmeans%s.mat', DATAopts.vocabularyPath, DescParam2Name(descParam))


% Test if Gaussian Mixture Model exists. If it does, return
if exist(vocabularySaveName, 'file')
    if ~exist('forceCreate', 'var') || forceCreate == 0
        % It already exists. Load and exit function
        load(vocabularySaveName);
        return
    end
end

% Get set of 1 million random descriptors
if ~exist('numDescriptors', 'var')
    numDescriptors = 1000000;
end
[descriptors, info] = GetRandomDescriptors(imNames, descParam, numDescriptors);

% pcaMap
if (descParam.pcaDim > size(descriptors,2)) || (descParam.pcaDim == 0)
    pcaMap.data.rot = 1;
    warning('descParam.pcaDim does not decrease dimensionality. Skipping PCA!!!');
else
    pcaMap = pca(descriptors(1:2:end,:), descParam.pcaDim);
end

% Descriptors to train Kmeans vocabulary, note these are disjunct from pca
descriptors = descriptors(2:2:end,:) * pcaMap.data.rot;



cell_Clusters=cell(1, size(descParam.Clusters, 2));

for i=1: size(descParam.Clusters, 2)
    
    [labs, ~, vocab] = kmeansj(descriptors, descParam.Clusters(i), 100);
    
    par.st_d=zeros(size(vocab));
    par.skew=zeros(size(vocab));
    par.kurt=zeros(size(vocab));
    par.nElem=zeros(size(vocab, 1), 1);  
    for j=1: size(vocab, 1)
        par.st_d(j, :)=std(descriptors(labs==j, :), 1);
        par.skew(j, :)=skewness(descriptors(labs==j, :));
        par.kurt(j, :)=kurtosis(descriptors(labs==j, :));
        par.nElem(j)=sum(labs==j);
    end
    
    par.vocabulary=vocab;
    
    cell_Clusters{i}=par;
    
end



cell_spClusters=cell(1, size(descParam.spClusters, 2));
spInfo=info.infoTraj(:, 8:9);

for i=1: size(descParam.spClusters, 2)
    
    [labs, ~, vocab] = kmeansj(spInfo, descParam.spClusters(i), 100);
    
    par.st_d=zeros(size(vocab));
    par.skew=zeros(size(vocab));
    par.kurt=zeros(size(vocab));
    par.nElem=zeros(size(vocab, 1), 1);  
    for j=1: size(vocab, 1)
        par.st_d(j, :)=std(spInfo(labs==j, :), 1);
        par.skew(j, :)=skewness(spInfo(labs==j, :));
        par.kurt(j, :)=kurtosis(spInfo(labs==j, :));
        par.nElem(j)=sum(labs==j);
    end
    
    par.vocabulary=vocab;
    
    cell_spClusters{i}=par;
    
end




save(vocabularySaveName, '-v7.3', 'pcaMap', 'cell_Clusters', 'cell_spClusters');