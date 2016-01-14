function [ featuresVideo ] = deL_noPoolFeatureMaps( pathVideo, layer )

featuresVideo=[];

switch layer
    
     case 'conv5_3' 
         
        dimFeatures=14*14*512;
        fileID=fopen(pathVideo); %open the file
        [featMaps, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer
        featMaps=featMaps';
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID);
        
        

        
        
        for i=1:14*14
            featuresVideo=cat(1,featuresVideo,featMaps(:, i:14*14:end));
        end
        
        
    otherwise
       fprintf('Unkonwn parameter!!!! the layer should be fc8, fc7, fc6 conv5_3 ...'); 

end

