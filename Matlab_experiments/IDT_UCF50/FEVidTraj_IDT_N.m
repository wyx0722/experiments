function [desc, info, descParam] = FEVidTraj_IDT_N(featurePath, descParam)

info.row=0;
info.col=0;
info.depth=0;
info.descSize=0;
info.vidSize=0;
info.imSize=0;
info.mediaName=featurePath;

[desc] = fast_descIDT(featurePath, descParam.IDTfeature);