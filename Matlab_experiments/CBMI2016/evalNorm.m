function [ ] = evalNorm( typeFeature )


alpha=0.1:0.1:0.9;

for i=1:length(alpha)
    
    FisherFramework(typeFeature, 'PNL2', alpha(i), 72, 256);
    VLADFramework(typeFeature, 'PNL2', alpha(i), 72, 512);
    
    
end


end

