function [vocabulary, pcaMap, st_d, skew, nElem, kurt] = CreateVocabularyKmeansPca_m(imNames, descParam, numClusters, pcaDim, ...
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

vocabularySaveName = sprintf('%sKmeans%sPca%gClusters%g.mat', DATAopts.vocabularyPath, ...
                       DescParam2Name(descParam), pcaDim, numClusters)


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
descriptors = GetRandomDescriptors(imNames, descParam, numDescriptors);

% pcaMap
if (pcaDim > size(descriptors,2)) || (pcaDim == 0)
    pcaMap.data.rot = 1;
    warning('pcaDim does not decrease dimensionality. Skipping PCA!!!');
else
    pcaMap = pca(descriptors(1:2:end,:), pcaDim);
end

% Descriptors to train Kmeans vocabulary, note these are disjunct from pca
descriptors = descriptors(2:2:end,:) * pcaMap.data.rot;

% First perform kmeans as initialization to gmm (max iter = 100)
[assign, ~, vocabulary] = kmeansj(descriptors, numClusters, 100);

st_d=zeros(size(vocabulary));
skew=zeros(size(vocabulary));
kurt=zeros(size(vocabulary));
nElem=zeros(size(vocabulary, 1), 1);
for i=1: size(vocabulary, 1)
    
    st_d(i, :)=std(descriptors(assign==i, :), 1);
    skew(i, :)=skewness(descriptors(assign==i, :));
    kurt(i, :)=kurtosis(descriptors(assign==i, :));
    nElem(i)=sum(assign==i);
    
end



save(vocabularySaveName, '-v7.3', 'pcaMap', 'vocabulary', 'st_d', 'skew', 'nElem', 'kurt');