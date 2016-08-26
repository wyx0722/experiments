

typeFeature=cell(1,2);
typeFeature{1}=@FEVidMBHxDense;
typeFeature{2}=@FEVidMBHyDense;


all_accuracy=cell(1,2);
all_clfsOut=cell(1,2);


normStrategy='ROOTSIFT';
d=72;
cl_VLAD=[256 512];
cl_FV=256;
fsr=1;
nPar=5;


for i=1:length(typeFeature)
    [all_accuracy{i}, all_clfsOut{i}]  = denseFramework(typeFeature{i}, normStrategy, d, cl_VLAD, cl_FV, fsr, nPar);    
end