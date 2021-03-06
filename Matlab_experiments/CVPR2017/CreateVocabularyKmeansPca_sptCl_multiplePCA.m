function [cell_Clusters, cell_spClusters, cell_pcaMap] = CreateVocabularyKmeansPca_sptCl_multiplePCA(imNames, descParam, ...
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



cell_Clusters=cell(length(cell_pcaMap), size(descParam.Clusters, 2));


for p=1:length(cell_pcaMap)
    
    %pcaMap=cell_pcaMap{p};
    % Descriptors to train Kmeans vocabulary, note these are disjunct from pca
    pca_descriptors = descriptors(2:2:end,:) * cell_pcaMap{p}.data.rot;





    for i=1: size(descParam.Clusters, 2)

        [labs, ~, vocab] = kmeansj(pca_descriptors, descParam.Clusters(i), 100);

        par.st_d=zeros(size(vocab));
        par.skew=zeros(size(vocab));
        par.kurt=zeros(size(vocab));
        par.nElem=zeros(size(vocab, 1), 1);  
        for j=1: size(vocab, 1)
            par.st_d(j, :)=std(pca_descriptors(labs==j, :), 1);
            par.skew(j, :)=skewness(pca_descriptors(labs==j, :));
            par.kurt(j, :)=kurtosis(pca_descriptors(labs==j, :));
            par.nElem(j)=sum(labs==j);
        end

        par.vocabulary=vocab;

        cell_Clusters{p,i}=par;

    end
end


cell_spClusters=cell(1, size(descParam.spClusters, 2));

if  strfind(descParam.MediaType,'IDT')
    sptInfo=info.infoTraj(:, 8:10);
else if strfind(descParam.MediaType,'DeepF')
     sptInfo=info.spInfo; 
    end
end

for i=1: size(descParam.spClusters, 2)
    
    [labs, ~, vocab] = kmeansj(sptInfo, descParam.spClusters(i), 100);
    
    par.st_d=zeros(size(vocab));
    par.skew=zeros(size(vocab));
    par.kurt=zeros(size(vocab));
    par.nElem=zeros(size(vocab, 1), 1);  
    for j=1: size(vocab, 1)
        par.st_d(j, :)=std(sptInfo(labs==j, :), 1);
        par.skew(j, :)=skewness(sptInfo(labs==j, :));
        par.kurt(j, :)=kurtosis(sptInfo(labs==j, :));
        par.nElem(j)=sum(labs==j);
    end
    
    par.vocabulary=vocab;
    
    cell_spClusters{i}=par;
    
end




save(vocabularySaveName, '-v7.3', 'cell_pcaMap', 'cell_Clusters', 'cell_spClusters');