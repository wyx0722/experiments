function [  ] = iter_normEval( bPathFeatures, network, normFM )


func=@FEVid_deepFeatures;
mediaType='DeepF';
layer='pool5';
nCl=256;
pcaD=256;
nPar=8;
norm='ROOTSIFT';
alpha=1;
for i=1:length(normFM)
    i
    normEval_VLAD(func, mediaType, layer, network, norm, alpha, nCl, pcaD, bPathFeatures, normFM{i}, nPar)

end

