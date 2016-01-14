function [desc, info, descParam, nrFrames] = MediaName2Descriptor(mediaName, descParam, pcaMap)
% [desc, info] = MediaName2Descriptor(mediaName, descParam, pcaMap)
%
% Converts an media file (image/video) into descriptors using the settings of 
% descParam. descParam contains a function handle @FE* with corresponding fields 
% as defined in the specific @FEx function.
%
% The function DescParam2Name can create a unique string for the feature.
%
% The function handles different media types by specifying
% descParam.Type as any of {'Im' (default), 'Vid', 'Load'} to obtain 
% descriptors from images, videos, and a file (with e.g. externally extracted
% descriptors). 
%
% mediaName:           Name of the image/video
% descParam:           Parameters of feature extraction
%   .Func:             Function handle of desired feature extraction (@FE*)
%                      The necessary parameters are specified in this function
%   .MediaType:        {'Im', 'Vid', 'Load'} image (default)/video/load from file
%   .Normalisation:    (optional) String: 'ROOTSIFT' (default), 'L1', 'L2', 'None'
%   .NumScales:        (optional) Number of scales to extract descriptors (default: 1)
%   .ScaleResizeFactor:(optional) resize per scale. Default: sqrt(0.5)
%   .ColourType:       (optional) Colour type. See Image2ColourSpace (default: 'Rgb');
%   .ClampF:           (optional) Clamping factor. After normalization, values higher
%                      than clampF are put to clampF and then re-normalized.
% pcaMap (optional):   PCA-map. If specified, applies only the rotation part 
%                      of the PCA-map
%
% desc:                N x D matrix with N descriptors of D dimensions 
% info:                Information structure of descriptors. For local
%                      descriptors contains e.g. coordinates.
% descParam:           descriptor parameters with all default values filled-in
%                      (enables sanity check of input descParam (typos e.d.))
%
%               Jasper Uijlings - 2013

% Deals with optional pcaMap parameter
if nargin == 2
    pcaMap = 1;
end

if ~isfield(descParam, 'MediaType')
    descParam.MediaType = 'Im';
end
type = descParam.MediaType;

switch type
    case 'Im' % Extract from image (normal)
        image = ImageRead(mediaName);
        [desc, info, descParam] = Image2Descriptor(image, descParam, pcaMap);
    case 'Vid' % Extract from video
        if isfield(descParam, 'ColourSpace')
            video = VideoRead(mediaName, descParam.ColourSpace);
        else
            video = VideoRead(mediaName);
        end
        [desc, info, descParam] = Video2Descriptor(video, descParam, pcaMap);
        %%%%%%%%%%%Ionut
        nrFrames=size(video, 3);
        %%%%%%%%%%%%%%%
        info.imSize = info.vidSize; % Hack to make it work. Rename to mediaSize
    case 'Load' % Load from file
        [desc, info, descParam] = descParam.Func(mediaName, descParam);
        [desc, descParam] = DescriptorNormClampPca(desc, descParam, pcaMap);
        
        %%%%%%%%%%%%Ionut
    case 'IDT'      
        [desc, info, descParam] = Video2Descriptor(mediaName, descParam, pcaMap);
        %%%%%%%%%%%%%Ionut
            %%%%%%%%%%%%Ionut
    case 'DeepF'      
        [desc, info, descParam] = Video2Descriptor(mediaName, descParam, pcaMap);
        %%%%%%%%%%%%%Ionut
    otherwise
        error('Unknown way of extracting descriptors. Should be {im, vid, load})');
end

% Store the name in the info structure
info.mediaName = mediaName;
