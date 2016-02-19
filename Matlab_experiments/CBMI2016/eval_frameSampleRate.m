function [ ] = eval_frameSampleRate( typeFeature, encodingMethod )


fsr=[2 3 6];

for i=1:length(fsr)
    encodingMethod
    i
    fsr(i)
    
    
    switch encodingMethod
        case 'FV'
            encodingMethod
            FisherFramework(typeFeature, 'ROOTSIFT', 72, 256, fsr(i));
        case 'VLAD'
            
            encodingMethod
            VLADFramework(typeFeature, 'ROOTSIFT', 72, 512, fsr(i));
            
        otherwise
            warning('Unexpected parameter. Should be VLAD or FV');
    end
    
end


end

