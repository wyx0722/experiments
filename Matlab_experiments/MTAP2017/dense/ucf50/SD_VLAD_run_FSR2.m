

typeFeature=cell(1,5);
typeFeature{1}=@FEVidHogDense
typeFeature{2}=@FEVidHofDense
typeFeature{3}=@FEVidHmgDense
typeFeature{4}=@FEVidMBHxDense
typeFeature{5}=@FEVidMBHyDense

all_accuracy=cell(1,5);
all_clfsOut=cell(1,5);


normStrategy='ROOTSIFT';
d=72;
cl=256;
fsr=1;
nPar=5;
alpha=0.1;


for i=1:length(typeFeature)
    [all_accuracy{i}, all_clfsOut{i}]  = SD_VLADFramework(typeFeature{i}, normStrategy, d, cl, fsr, nPar, alpha);
end