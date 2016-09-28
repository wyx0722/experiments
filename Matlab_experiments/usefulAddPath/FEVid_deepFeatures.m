function [desc, info, descParam] = FEVid_deepFeatures(featurePath, descParam)

info.row=0;
info.col=0;
info.depth=0;
info.descSize=0;
info.vidSize=0;
info.imSize=0;
info.mediaName=featurePath;

if ~isfield(descParam, 'NormFeatureMaps')
    descParam.NormFeatureMaps='None';
end

if strfind(descParam.net, 'C3D')>0
    [desc, spInfo] = maps2features_c3d(featurePath, descParam.Layer);
else
    [desc, spInfo] = maps2features(featurePath, descParam.Layer, descParam.NormFeatureMaps);

end

info.spInfo=spInfo;