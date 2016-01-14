function [ featuresVideo ] = tempPoolFeatureMaps_slow( pathVideo, layer, nPool )

featuresVideo=[];

switch layer
    
     case 'conv5_3' 
         tic
        dimFeatures=14*14*512;
        fileID=fopen(pathVideo); %open the file
        [featMaps, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer
        featMaps=featMaps';
        nFeatursMaps=size(featMaps,1);
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID);
        toc
        
        tic
        if nPool>1
            modPool=mod(size(featMaps, 1),nPool);
            
            if modPool>(nPool/2 -1)
                poolFeatMaps=zeros(floor(nFeatursMaps/nPool)+1, dimFeatures);
                poolFeatMaps(end, :)=sum(featMaps(end-nPool+1:end, :), 1);
            else
                poolFeatMaps=zeros(floor(nFeatursMaps/nPool), dimFeatures);
            end
            
            k=1;
            for i=nPool:nPool:size(featMaps, 1)
                poolFeatMaps(k, :)=sum(featMaps(i-nPool+1:i, :), 1);
                k=k+1;
            end
            
        else
            poolFeatMaps=featMaps;
        end
        toc
        
        tic
        for i=1:14*14
            featuresVideo=cat(1,featuresVideo,poolFeatMaps(:, i:14*14:end));
        end
        toc
        
    otherwise
       fprintf('Unkonwn parameter!!!! the layer should be fc8, fc7, fc6 conv5_3 ...'); 

end

