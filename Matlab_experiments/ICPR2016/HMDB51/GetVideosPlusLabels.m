% Get videos, labels, and groups
%
% Groups is a N x 3 vector with 3 splits, where (as per official instructions):
%   1 denotes training set
%   2 denotes testing set
%   0 denotes not included in this plit
function [vids labs groups] = GetVideosPlusLabels(class)

global DATAopts;

if nargin == 0
    theClasses = 1:DATAopts.nclasses;
else
    theClasses = class;
end

allVids = cell(DATAopts.nclasses,1);
allGroups = cell(DATAopts.nclasses,1);
allLabs = cell(DATAopts.nclasses,1);
for cI=theClasses
    % Get videos and splits through predefined splits
    splitName = [DATAopts.datadir 'Splits/' DATAopts.classes{cI} '_test_split%d.txt'];

    ff = fopen(sprintf(splitName, 1));
    data = textscan(ff, '%s%d');
    fclose(ff);

    videoNames = data{1};
    splitNrs{1} = data{2};

    ff = fopen(sprintf(splitName, 2));
    data = textscan(ff, '%s%d');
    fclose(ff);

    if ~isequal(videoNames, data{1})
        keyboard;
    end

    videoNames = data{1};
    splitNrs{2} = data{2};

    ff = fopen(sprintf(splitName, 3));
    data = textscan(ff, '%s%d');
    fclose(ff);

    if ~isequal(videoNames, data{1})
        keyboard;
    end

    videoNames = data{1};
    splitNrs{3} = data{2};

    theSplitNrs = cat(2, splitNrs{:});

    theSplitNrs = double(theSplitNrs);
    %%% End of obtaining splits
    
    % Add the class path to the video name
    for i=1:length(videoNames)
        videoNames{i} = [DATAopts.classes{cI} '/' videoNames{i}];
    end
    
    allVids{cI} = videoNames;
    allGroups{cI} = theSplitNrs;
    allLabs{cI} = zeros(length(videoNames), DATAopts.nclasses);
    allLabs{cI}(:,cI) = 1;
end

vids = cat(1, allVids{:});
labs = cat(1, allLabs{:});
groups = cat(1, allGroups{:});
