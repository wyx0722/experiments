
typeFeature=cell(1,2);
typeFeature{1}=@FEVidMBHxDense
typeFeature{2}=@FEVidMBHyDense


all_accuracy=cell(1,2);
all_clfsOut=cell(1,2);


normStrategy='ROOTSIFT';
d=72;
cl=256;
fsr=1;
nPar=5;
alpha=0.5;



    [all_accuracy{2}, all_clfsOut{2}]  = FisherFramework(typeFeature{2}, normStrategy, d, cl, fsr, nPar, alpha);




% typeFeature=cell(1,2);
% typeFeature{1}=@FEVidMBHxDense
% typeFeature{2}=@FEVidMBHyDense
% 
% 
% all_accuracy=cell(1,2);
% all_clfsOut=cell(1,2);
% 
% 
% normStrategy='ROOTSIFT';
% d=72;
% cl=256;
% fsr=1;
% nPar=5;
% alpha=0.5;
% 
% 
% for i=1:length(typeFeature)
%     [all_accuracy{i}, all_clfsOut{i}]  = FisherFramework(typeFeature{i}, normStrategy, d, cl, fsr, nPar, alpha);
% end