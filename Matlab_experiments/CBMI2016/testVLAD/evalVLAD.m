function [ ] = evalVLAD( typeFeature )


    VLADFramework_noMean(typeFeature, 'ROOTSIFT', 1, 72, 512);
    VLADFramework_mean(typeFeature, 'ROOTSIFT', 1, 72, 512);
    VLADFramework_mean_fast(typeFeature, 'ROOTSIFT', 1, 72, 512);
    


end

