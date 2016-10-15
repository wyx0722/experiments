function [vids, labs, groups, testI] = GetVideosPlusLabels(maxGroup)
% [vids, labs, groups] = GetVideosPlusLabels(maxGroup)
%
% Get the videos, labels, and also group numbers for original UCF50
% cross-validation datasets.
%
% maxGroup (optional):  max group number OR 'small' (maxGroup = 7)
%
% vids:                 N x 1 cell array with Video names
% labs:                 N x C Label array with C classes
% groups:               N x 1 vector with group numbers for leave-one-out
%                       crossvalidation
%
%               Jasper Uijlings - 2013

global DATAopts;

allVids = cell(DATAopts.nclasses,1);
allGroups = cell(DATAopts.nclasses,1);
allLabs = cell(DATAopts.nclasses,1);
for cI=1:DATAopts.nclasses;
    nameStruct = dir(sprintf(DATAopts.videoPath, [DATAopts.classes{cI} '/*']));
    
    % Get individual classes
    classVids = cell(length(nameStruct),1);
    classGroups = zeros(length(nameStruct),1);
    
    for i=1:length(nameStruct)
        classVids{i} = [DATAopts.classes{cI} '/' nameStruct(i).name(1:end-4)];
        
        % Get group nr
        [startI, endI] = regexp(nameStruct(i).name, '_g\d+_');
        classGroups(i) = str2double(nameStruct(i).name(startI+2:endI-1));
    end
    
    allVids{cI} = classVids;
    allGroups{cI} = classGroups;
    allLabs{cI} = zeros(length(nameStruct), DATAopts.nclasses);
    allLabs{cI}(:,cI) = 1;
end

vids = cat(1, allVids{:});
labs = cat(1, allLabs{:});
groups = cat(1, allGroups{:});

% Get the normal set or the small set
if nargin < 1
        % There are 25 groups in the official test settings, although more
    % groups are in the actual dataset (but these have too few examples)
    maxGroup = 25; 
else
    switch maxGroup
        case 'small'
            maxGroup = 7;
        case 'smallEnd'
            groups = mod(groups+12, 30) + 1;
            maxGroup = 7;
        case 'Ionut'
            % Take half the dataset
            vids = vids(1:3320);
            labs = labs(1:3320,:);
            groups = groups(1:3320);
            
            goodI = groups < 26;
            groups = ceil(groups ./ 5);
            [~, allLabsVector] = find(labs);
            goodI = goodI & (allLabsVector <= 20);
            groups = groups(goodI);
            labs = labs(goodI,1:20);
            return;
        case 'Full'
            maxGroup = 25; % Full set
        case 'Challenge'
            trainI = zeros(length(vids), 3);
            testI = zeros(length(vids), 3);

            for foldI=1:3

                [trainIds, ~] = textread(sprintf('%s/ucfTrainTestlist/trainlist0%d.txt', DATAopts.datadir, foldI), '%s %f');
                testIds = textread(sprintf('%s/ucfTrainTestlist/testlist0%d.txt', DATAopts.datadir, foldI), '%s');
                for tI = 1:length(trainIds)
                    trainIds{tI} = trainIds{tI}(1:end-4);
                end
                for tI = 1:length(testIds)
                    testIds{tI} = testIds{tI}(1:end-4);
                end

                trainI(:,foldI) = ismember(vids, trainIds);
                testI(:,foldI) = ismember(vids, testIds);
            end
            
            % Make logical arrays
            trainI = trainI > 0;
            testI = testI > 0;
            
            groups = trainI;
            return;
        otherwise
            error('Wrong input argument\n');
    end
end

goodI = groups <= maxGroup;
vids = vids(goodI);
labs = labs(goodI,:);
groups = groups(goodI);
