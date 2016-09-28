function [ featuresVideo, spInfo ] = maps2features_c3d( pathFeaturesVideo, layer)

%pathFeaturesVideo - represents the file where the deep features are saved



switch layer
    
        
    case 'pool5'
        rows=4;
        cols=4;
        channels=512;
        
        load(pathFeaturesVideo); %allFeatures
        
        
        featuresVideo=zeros(rows*cols*length(allFeatures), channels);
        spInfo=zeros(rows*cols*length(allFeatures), 3);
        p=1;

        for i=1:length(allFeatures)
            
            for m=1:size(allFeatures{i}, 4)
                for n=1:size(allFeatures{i}, 5)
                   featuresVideo(p, :)=allFeatures{i}(1, :, 1, m, n);
                   spInfo(p, 1)=m/rows;
                   spInfo(p, 2)=n/cols;
                   spInfo(p, 3)=i/length(allFeatures);
                   p=p+1;
                end
            end
            
        end

        
    otherwise
       error('Unkonwn parameter layer: %s ', layer); 
end
 

end

