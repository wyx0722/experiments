function [descriptors, info] = GetRandomDescriptors(imNames, descParam, numD)
% descriptors = GetRandomDescriptors(imNames, descParam, numD)
%
% Returns a random set of descriptors
%
% imNames:       List of imageNames to get Random Descriptors from
% descParam:    Type and settings for descritor extraction
% numD:         number of random descriptors to be returned
%
% descriptors:  Random set of descriptors. Row-wise

% Make fake labels to be able to re-use GetRandomLabeledDescriptors
fakeLabels = ones(length(imNames),1);

% Get random descriptors
[descriptors, ~, info] = GetRandomLabeledDescriptors(imNames, fakeLabels, descParam, numD);
