function [  ] = iter_normEval( bPathFeatures, network, norm, alpha )


func=@FEVid_deepFeatures;
mediaType='DeepF';
layer='pool5';
nCl=256;
pcaD=256;
nPar=10;

for i=1:length(norm)
    i
    normEval_VLAD(func, mediaType, layer, network, norm{i}, alpha(i), nCl, pcaD, bPathFeatures, nPar)

end

