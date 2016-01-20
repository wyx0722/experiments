function [ intra_features ] = intranormalizationFeatures( features, initDim )


intra_features=zeros(size(features));

for i=1:initDim:size(features, 2)
    intra_features(:, i:i+initDim-1)=NormalizeRowsUnit(features(:, i:i+initDim-1));    
end


end

