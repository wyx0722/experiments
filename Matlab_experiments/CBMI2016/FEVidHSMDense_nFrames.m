function [desc, info, descParam] = FEVidHSMDense_nFrames(video, descParam)
% [desc, info, descParam] = FEVidHSMDense_nFrames(video, descParam)
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

%!!!!compute motion: (F1-F2) + (F1-F3) + (F1-F4) ....

simpleMotion_nF = video(:, :, 1:end-descParam.nFrames+1) - video(:, :, 2:end-descParam.nFrames+2);
for i=2:descParam.nFrames-1

        simpleMotion_nF = simpleMotion_nF + (video(:, :, 1:end-descParam.nFrames+1) - video(:, :, i+1:end+(i-descParam.nFrames+1)));
end
    

[desc, info] = Video2DenseHOGVolumes(simpleMotion_nF, ...
                                     descParam.BlockSize, ...
                                     descParam.NumBlocks, ...
                                     descParam.NumOr);