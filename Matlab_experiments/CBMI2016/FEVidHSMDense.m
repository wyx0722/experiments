function [desc, info, descParam] = FEVidHSMDense(video, descParam)
% [desc, info, descParam] = FEVidHSMDense(video, descParam)
%
% Wrapper to densely sample HSM features from a video using the 
% fast Video2DenseHOGVolumes function
%
% video:            N x M x F grayscale video
% descParam
%   .BlockSize:     1 x 3 vector with number of pixels per subblock [row col frame]
%   .NumBlocks:     1 x 3 vector with number of subblock per direction [row col frame]
%   .NumOr:         Number of orientations per HOG descriptor
%
% desc:             T x (numBlocks(1) * numBlocks(2) * numBlocks(3) * numOr)
%                   matrix with T row-wise features
% info:             Corresponding information structure
%
%       Ionut-Cosmin Duta - 2016

% simpleMotion=video(:, :, 1:end-1)-video(:, :, 2:end);
% 
% [desc, info] = Video2DenseHOGVolumes(simpleMotion, ...
%                                      descParam.BlockSize, ...
%                                      descParam.NumBlocks, ...
%                                      descParam.NumOr);

%simpleMotion=video(:, :, 1:end-1)-video(:, :, 2:end);

[desc, info] = Video2DenseHOGVolumes(video(:, :, 1:end-1)-video(:, :, 2:end), ...
                                     descParam.BlockSize, ...
                                     descParam.NumBlocks, ...
                                     descParam.NumOr);