function [desc, info, descParam] = FEVid_deepFeatures_spInfoNorm4(featurePath, descParam)

info.row=0;
info.col=0;
info.depth=0;
info.descSize=0;
info.vidSize=0;
info.imSize=0;
info.mediaName=featurePath;

[desc, spInfo] = maps2features(featurePath, descParam.Layer);

maxV=max(max(desc));
desc=cat(2, desc, (spInfo(:, 1)./320).*maxV, (spInfo(:, 2)./240).*maxV);



info.spinfo=spInfo;