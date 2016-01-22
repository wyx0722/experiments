function [desc, descParam] = DescriptorNormClampPca(desc, descParam, pcaMap)
% desc = DescriptorNormClampPca(desc, descParam, pcaMap)
%
% Performs normalisation, clamping, and PCA on descriptors as a
% post-processing step. Clamping is the cutting of high values in a
% descriptor.
%
% desc:             N x D matrix of N descriptors of D dimensions
% descParam:
%   .Normalisation: Normalisation strategy. Default: ROOTSIFT. Possible:
%                   {'ROOTSIFT', 'L2', 'L1', 'None'}
%   .ClampF:        Clamping Factor. After normalization, all values above 
%                   clampF are put to clampF, after which descriptors are 
%                   renormalized.
% pcaMap:           PCA map with the rotation matrix
%
% desc:             N x P matrix of N descriptors of P dimensions, P is
%                   determined by pcaMap
% descParam:        descriptor parameters with default values filled in.
%
%           Jasper Uijlings - 2013

%%% Normalization
if ~isfield(descParam, 'Normalisation')
    descParam.Normalisation = 'ROOTSIFT';
end
normStrategy = descParam.Normalisation;

switch normStrategy
    case 'ROOTSIFT' % L1 -> Sqrt
        desc = sqrt(NormalizeRows(desc));
    case 'L2'
        desc = NormalizeRowsUnit(desc);
    case 'L1'
        desc = NormalizeRows(desc);
        
        %%%Ionut
    case 'PN'
        desc=PowerNormalization(desc, descParam.alpha);
    case 'disp_traj'
        desc=norm_disp_traj(desc);
    case 'L1PN' % L1 -> PN
        desc=PowerNormalization(NormalizeRows(desc), descParam.alpha);
    case 'PNL2'% PN -> L2
        desc= NormalizeRowsUnit(PowerNormalization(desc, descParam.alpha));
        %%%%%%%
        
    case 'None'
        % No normalization happens here
    otherwise
        warning('No valid normalization strategy');
end

%%% Clamping
if isfield(descParam,'ClampF')
    % Perform clamping
    clampF = descParam.ClampF;
    desc(desc > clampF) = clampF;
    
    % Renormalize after clamping
    switch normStrategy
        case 'ROOTSIFT' % Sqrt has already been done
            desc = NormalizeRowsUnit(desc);
        case 'L2'
            desc = NormalizeRowsUnit(desc);
        case 'L1'
            desc = NormalizeRows(desc);
        case 'None'
            % No normalization happens here
        otherwise
            warning('No valid normalization strategy');
    end
end

%%% PCA
% Perform PCA if there is a pcaMap field
if exist('pcaMap', 'var')
    if isa(pcaMap, 'mapping') || isstruct(pcaMap)
        desc = desc * pcaMap.data.rot;
    else
        desc = desc * pcaMap;
    end
end
