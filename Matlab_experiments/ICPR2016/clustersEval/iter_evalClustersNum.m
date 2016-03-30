function [  ] = iter_evalClustersNum( bPathFeatures, network, clustesDim )


func=@FEVid_deepFeatures;
mediaType='DeepF';
layer='pool5';
norm='ROOTSIFT';
dimPCA=256;
nPar=5;

for i=1:length(clustesDim)
    i
    evalClustersNum_VLAD(func, mediaType, layer, network, norm, clustesDim(i), dimPCA, bPathFeatures, nPar)
   

end

