function [pcaMap, orgCluster, bovwCluster, cell_smallCls, bigCluster] = CreateVocabularyKmeansPca_sepVocab(imNames, descParam, numClusters_original, nCl_BOVW, nClSmall, pcaDim, ...
                                                         sizeBigCluster, numDescriptors, forceCreate)
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

vocabularySaveName = sprintf('%sKmeans%sPca%gClusters%gnCl_BOVW%gnClSmall%g.mat', DATAopts.vocabularyPath, ...
                       DescParam2Name(descParam), pcaDim, numClusters_original, nCl_BOVW, nClSmall)


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
[assign, ~, vocabulary] = kmeansj(descriptors, numClusters_original, 100);

orgCluster.st_d=zeros(size(vocabulary));
orgCluster.skew=zeros(size(vocabulary));
orgCluster.kurt=zeros(size(vocabulary));
orgCluster.nElem=zeros(size(vocabulary, 1), 1);
for i=1: size(vocabulary, 1)
    orgCluster.st_d(i, :)=std(descriptors(assign==i, :), 1);
    orgCluster.skew(i, :)=skewness(descriptors(assign==i, :));
    orgCluster.kurt(i, :)=kurtosis(descriptors(assign==i, :));
    orgCluster.nElem(i)=sum(assign==i);   
end
orgCluster.vocabulary=vocabulary;

%for big vocabulary
if ~exist('sizeBigCluster', 'var')
    sizeBigCluster = 512;
end

[assign, ~, vocabulary] = kmeansj(descriptors, sizeBigCluster, 100);

bigCluster.st_d=zeros(size(vocabulary));
bigCluster.skew=zeros(size(vocabulary));
bigCluster.kurt=zeros(size(vocabulary));
bigCluster.nElem=zeros(size(vocabulary, 1), 1);
for i=1: size(vocabulary, 1)
    bigCluster.st_d(i, :)=std(descriptors(assign==i, :), 1);
    bigCluster.skew(i, :)=skewness(descriptors(assign==i, :));
    bigCluster.kurt(i, :)=kurtosis(descriptors(assign==i, :));
    bigCluster.nElem(i)=sum(assign==i);   
end
bigCluster.vocabulary=vocabulary;

%for the BoVW cluster
[assign, ~, vocabulary] = kmeansj(descriptors, nCl_BOVW, 100);

bovwCluster.st_d=zeros(size(vocabulary));
bovwCluster.skew=zeros(size(vocabulary));
bovwCluster.kurt=zeros(size(vocabulary));
bovwCluster.nElem=zeros(size(vocabulary, 1), 1);
for i=1: size(vocabulary, 1)    
    bovwCluster.st_d(i, :)=std(descriptors(assign==i, :), 1);
    bovwCluster.skew(i, :)=skewness(descriptors(assign==i, :));
    bovwCluster.kurt(i, :)=kurtosis(descriptors(assign==i, :));
    bovwCluster.nElem(i)=sum(assign==i);    
end
bovwCluster.vocabulary=vocabulary;


%create all the clusters
cell_smallCls=cell(1,size(bovwCluster.vocabulary, 1));

for i=1:size(bovwCluster.vocabulary, 1)
    
    tDesc=descriptors(assign==i, :);
    
    [labs, ~, vocab] = kmeansj(tDesc, nClSmall, 100);
    
    par.st_d=zeros(size(vocab));
    par.skew=zeros(size(vocab));
    par.kurt=zeros(size(vocab));
    par.nElem=zeros(size(vocab, 1), 1);  
    for j=1: size(vocab, 1)
        par.st_d(j, :)=std(tDesc(labs==j, :), 1);
        par.skew(j, :)=skewness(tDesc(labs==j, :));
        par.kurt(j, :)=kurtosis(tDesc(labs==j, :));
        par.nElem(j)=sum(labs==j);
    end
    
    par.vocabulary=vocab;
    
    cell_smallCls{i}=par;
    
end
    


save(vocabularySaveName, '-v7.3', 'pcaMap', 'orgCluster', 'bovwCluster', 'cell_smallCls');