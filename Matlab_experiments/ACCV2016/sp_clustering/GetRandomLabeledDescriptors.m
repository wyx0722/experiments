function [desc, descLabs, info] = GetRandomLabeledDescriptors(imNames, imLabs, descParam, numD)
% [desc, labs] = GetRandomLabeledDescriptors(imNames, descParam, numD)
%
% Obtain numD descriptors which are labeled on image level, randomly
% sampled from the images in imNames
%
% imNames:          N x 1 List of image names
% imLabs:           N x C Label matrix for each image for each class C
% descParam:        Type and Settings for descriptor extraction
% numD:             Total number of random descriptors to extract
%
% desc:             List of descriptors
% labs:             Labels for each descriptor.

% Get the number of descriptors per instance of each class
numClasses = size(imLabs, 2);
descriptorsPerClass = floor(numD / numClasses);
numExamples = sum(imLabs);
numPerInstance = ceil(descriptorsPerClass ./ numExamples);

% Get random descriptors per class
descI = 1;
fprintf('Getting random descriptors per class\n%d images: ', length(imNames));
for i=1:length(imNames)
    if mod(i,100) == 0
        fprintf('%d ', i);
    end
    
    % Extract descriptors from image
    [currDesc, currInfo] = MediaName2Descriptor(imNames{i}, descParam); %Ionut!!!!!!!!!!!!!
    
    descT = cell(1, numClasses);
    infoT = cell(1, numClasses); %ionut!!!
    descLabsT = cell(1, numClasses);
    numDescT = 0;
    for cI=1:numClasses
        if imLabs(i,cI)
            % Get random descriptors
            randI = randperm(size(currDesc,1));
            numI = min(length(randI), numPerInstance(cI));
            
            descT{cI} = currDesc(randI(1:numI),:);
            infoT{cI} = currInfo.infoTraj(randI(1:numI),:);%Ionut!!!!
            
            currLabs = zeros(numI,numClasses);
            currLabs(:,cI) = 1;
            descLabsT{cI} = currLabs;
            numDescT = numDescT + numI; % Number of descriptors for this image
        end
    end
    
    % Memory allocation desc, descLabs
    if i == 1
        descTemp = cat(1, descT{:});
        totalNumDescriptors = sum(numPerInstance .* numExamples);
        desc = zeros(totalNumDescriptors, size(descTemp,2));
        descLabs = zeros(totalNumDescriptors, numClasses);
        info.infoTraj=zeros(totalNumDescriptors, size(currInfo.infoTraj,2)); %Ionut
    end
    
    % Add random descriptors to the final set
    if numDescT > 0
        iB = descI;
        iE = descI + numDescT - 1;
        desc(iB:iE,:) = cat(1, descT{:});
        descLabs(iB:iE,:) = cat(1, descLabsT{:});
        info.infoTraj(iB:iE,:)=cat(1, infoT{:}); %Ionut!!!!
        descI = iE+1;
    end
end
fprintf('\n');

% Delete overallocated memory
desc = desc(1:descI-1,:);
descLabs = descLabs(1:descI-1,:);
info.infoTraj = info.infoTraj(1:descI-1,:); %Ionut!!!
