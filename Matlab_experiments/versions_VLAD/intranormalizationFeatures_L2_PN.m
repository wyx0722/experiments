function [ intra_features ] = intranormalizationFeatures_L2_PN( features, initDim, alpha )


intra_features=zeros(size(features));

for i=1:initDim:size(features, 2)
    intra_features(:, i:i+initDim-1)=NormalizeRowsUnit(PowerNormalization(features(:, i:i+initDim-1), alpha));    
end


end

