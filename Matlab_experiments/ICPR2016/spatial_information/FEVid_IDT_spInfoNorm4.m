function [desc, info, descParam] = FEVid_IDT_spInfoNorm4(featurePath, descParam)

info.row=0;
info.col=0;
info.depth=0;
info.descSize=0;
info.vidSize=0;
info.imSize=0;
info.mediaName=featurePath;

maxValueHog=0.76;

[desc, iTraj] = fast_descIDT(featurePath, descParam.IDTfeature);
desc=cat(2, desc, (iTraj(:, 2)./320).*maxValueHog, (iTraj(:, 3)./240).*maxValueHog);

info.infoTraj=iTraj;