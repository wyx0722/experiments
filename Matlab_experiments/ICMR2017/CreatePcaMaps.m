function [cell_pcaMap] = CreatePcaMaps(imNames, descParam, numDescriptors, forceCreate)
                                                         
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

pcaMaps_SaveName = sprintf('%sPcaMaps__%s.mat', DATAopts.vocabularyPath, DescParam2Name(descParam))


% Test if Gaussian Mixture Model exists. If it does, return
if exist(pcaMaps_SaveName, 'file')
    if ~exist('forceCreate', 'var') || forceCreate == 0
        % It already exists. Load and exit function
        load(pcaMaps_SaveName);
        return
    end
end

% Get set of 1 million random descriptors
if ~exist('numDescriptors', 'var')
    numDescriptors = 1000000;
end
[descriptors, info] = GetRandomDescriptors(imNames, descParam, numDescriptors);

% pcaMap

cell_pcaMap=cell(1, length(descParam.pcaDim));
for p=1:length(descParam.pcaDim)
    if (descParam.pcaDim(p) > size(descriptors,2)) || (descParam.pcaDim(p) == 0)
        pcaMap.data.rot = 1;
        warning('descParam.pcaDim does not decrease dimensionality. Skipping PCA!!!');
        cell_pcaMap{p}=pcaMap;
    else
        cell_pcaMap{p} = pca(descriptors(1:2:end,:), descParam.pcaDim(p));
        %cell_pcaMap{p}=pcaMap;
    end
end




save(pcaMaps_SaveName, '-v7.3', 'cell_pcaMap');