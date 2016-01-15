function [desc, info, descParam] = FEVid_conv5_3_overlapTempPool(featurePath, descParam)

info.row=0;
info.col=0;
info.depth=0;
info.descSize=0;
info.vidSize=0;
info.imSize=0;
info.mediaName=featurePath;

[desc] = OverlapTempPoolFeatureMaps(featurePath, descParam.Layer, descParam.nPool, descParam.overlap);
