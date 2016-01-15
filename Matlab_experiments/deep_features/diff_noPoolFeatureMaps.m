function [ featuresVideo ] = diff_noPoolFeatureMaps( pathVideo, layer )

%featuresVideo=[];

switch layer
    
     case 'conv5_3' 
         %tic
        
        dimConvMap=14*14;
        channels=512;
        dimFeatures=dimConvMap * channels;
        fileID=fopen(pathVideo); %open the file
        [featMaps, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer
        featMaps = featMaps';
        featMaps = featMaps(1:end-1, :) - featMaps(2:end, :);
        nFeatursMaps=size(featMaps,1);
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID);
        %toc
        

        
        %tic
        featuresVideo=zeros(dimConvMap * nFeatursMaps, channels);
        step=nFeatursMaps;
        for i=1:dimConvMap
            featuresVideo(i*step-step+1:i*step,:)=featMaps(:, i:dimConvMap:end);
        end
       % toc
       
       case 'pool5' 
         %tic
        
        dimConvMap=7*7;
        channels=512;
        dimFeatures=dimConvMap * channels;
        fileID=fopen(pathVideo); %open the file
        [featMaps, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer
        featMaps = featMaps';
        featMaps = featMaps(1:end-1, :) - featMaps(2:end, :);
        nFeatursMaps=size(featMaps,1);
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID);
        %toc
        

        
        %tic
        featuresVideo=zeros(dimConvMap * nFeatursMaps, channels);
        step=nFeatursMaps;
        for i=1:dimConvMap
            featuresVideo(i*step-step+1:i*step,:)=featMaps(:, i:dimConvMap:end);
        end
       % toc
        
    otherwise
       fprintf('Unkonwn parameter!!!! the layer should be fc8, fc7, fc6 conv5_3 ...'); 

end

