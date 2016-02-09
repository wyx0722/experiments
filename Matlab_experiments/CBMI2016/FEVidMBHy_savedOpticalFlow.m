function [desc, info, descParam] = FEVidMBHy_savedOpticalFlow(opticalFlow, descParam)
% [desc, info] = FEVidHogDense(video, descParam)
%
% Wrapper to densely sample MBH (for horizontal direction) features from a
% video (optical flow of horizontal component) using the fast 
% Video2DenseHOGVolumes function
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
%       Ionut Cosmin Duta - 2014


yOpticalFlow=imag(opticalFlow);

[desc, info] = Video2DenseHOGVolumes(yOpticalFlow, ...
                                     descParam.BlockSize, ...
                                     descParam.NumBlocks, ...
                                     descParam.NumOr);