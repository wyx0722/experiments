function [  ] = iter_evaldimPCA( bPathFeatures, network, dimsPCA )


func=@FEVid_deepFeatures;
mediaType='DeepF';
layer='pool5';
norm='ROOTSIFT';
nCl=256;
dAssign=1;
nPar=8;

for i=1:length(dimsPCA)
    i
    evalPCAdim_VLAD(func, mediaType, layer, network, norm, nCl, dimsPCA(i), bPathFeatures, dAssign, nPar)

end

