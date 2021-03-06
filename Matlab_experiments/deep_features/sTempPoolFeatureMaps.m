function [ featuresVideo ] = sTempPoolFeatureMaps( pathVideo, layer, nPool )



switch layer
    
     case 'conv5_3' 
         
        dimConvMap=14*14;
        channels=512;
        dimFeatures=dimConvMap * channels;
        fileID=fopen(pathVideo); %open the file
        [featMaps, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer
        featMaps=featMaps';
        nFeatursMaps=size(featMaps,1);
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID);
        
        
        
        if nPool>1
            
            poolFeatMaps=zeros(nFeatursMaps-nPool+1, dimFeatures);
                      
            for i=1:nFeatursMaps-nPool+1
                poolFeatMaps(i, :)=sum(featMaps(i:i+nPool-1, :), 1);
                
            end
            
        else
            poolFeatMaps=featMaps;
        end
        
        
        
        nPoolFeatMaps=size(poolFeatMaps, 1);
        featuresVideo=zeros(dimConvMap * nPoolFeatMaps, channels);
        
        step=nPoolFeatMaps;
        for i=1:dimConvMap      
            featuresVideo(i*step-step+1:i*step,:)=poolFeatMaps(:, i:dimConvMap:end);
        end
        
        
    otherwise
       fprintf('Unkonwn parameter!!!! the layer should be fc8, fc7, fc6 conv5_3 ...'); 

end

