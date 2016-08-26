

typeFeature=cell(1,3);
typeFeature{1}=@FEVidHogDense;
typeFeature{2}=@FEVidHofDense;
typeFeature{3}=@FEVidHmgDense;

all_accuracy=cell(1,3);
all_clfsOut=cell(1,3);


normStrategy='ROOTSIFT';
d=72;
cl_VLAD=[256 512];
cl_FV=256;
fsr=1;
nPar=7;


for i=1:length(typeFeature)
    [all_accuracy{i}, all_clfsOut{i}]  = denseFramework(typeFeature{i}, normStrategy, d, cl_VLAD, cl_FV, fsr, nPar);    
end