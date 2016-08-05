function [vocabsClass, pcaMap] = CreateVocabularyKmeans_simVocabs_par(trainClass, descParam, nPar, ...
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
if ~exist('nPar', 'var')
    nPar = 1;
end


vocabsClass=cell(size(trainClass));

fprintf('Creating  %d vocabularies: \n', length(trainClass)); 
if nPar<=1
    for i=1:length(trainClass)
        fprintf('%d ', i);
        [vocabsClass{i}]=getVocabPCAClass(trainClass{i}, descParam, numDescriptors);    
    end
else
    parpool(nPar);
    parfor i=1:length(trainClass)
        fprintf('%d \n', i);
        [vocabsClass{i}]=getVocabPCAClass(trainClass{i}, descParam, numDescriptors);    
    end
    delete(gcp('nocreate'))   
end
fprintf('\nDone creating the vocabularies!\n');

pcaMap.data.rot = 1;


save(vocabularySaveName, '-v7.3', 'vocabsClass', 'pcaMap');
end


function [vocabClass]=getVocabPCAClass(trainClass, descParam, numDescriptors)


    [descriptors, info] = GetRandomDescriptors(trainClass, descParam, numDescriptors);
    
    [labs, ~, vocabClass] = kmeansj(descriptors, descParam.sizeVocab, 100);
    
    %!!!!!!!!!!
end