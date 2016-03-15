function [desc, info, descParam] = FEVid_IDT_spInfoNorm3(featurePath, descParam)

info.row=0;
info.col=0;
info.depth=0;
info.descSize=0;
info.vidSize=0;
info.imSize=0;
info.mediaName=featurePath;

[desc, iTraj] = fast_descIDT(featurePath, descParam.IDTfeature);
maxV=max(max(desc));
desc=cat(2, desc, (iTraj(:, 2)./320).*maxV, (iTraj(:, 3)./240).*maxV);

info.infoTraj=iTraj;