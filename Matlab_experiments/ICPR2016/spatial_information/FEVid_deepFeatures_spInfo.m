function [desc, info, descParam] = FEVid_deepFeatures_spInfo(featurePath, descParam)

info.row=0;
info.col=0;
info.depth=0;
info.descSize=0;
info.vidSize=0;
info.imSize=0;
info.mediaName=featurePath;

[desc, spInfo] = maps2features(featurePath, descParam.Layer);

desc=cat(2, desc, spInfo);


info.spinfo=spInfo;