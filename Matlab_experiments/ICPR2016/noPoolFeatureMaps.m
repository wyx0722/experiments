function [ featuresVideo, spInfo ] = noPoolFeatureMaps( pathVideo, layer )

%featuresVideo=[];
spInfo=[];
switch layer
    
     case 'conv5_3' 
         %tic
        
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
        featMaps=featMaps';
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

       case 'pool5_withSpInfo' 
        
        rows=7;
        cols=7;
        dimConvMap=rows*cols;
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
        
        

        
        
        featuresVideo=zeros(dimConvMap * nFeatursMaps, channels);
        step=nFeatursMaps;
        for i=1:dimConvMap
            featuresVideo(i*step-step+1:i*step,:)=featMaps(:, i:dimConvMap:end);
        end
        
        
        %get spatial informaton of the features: the pozition on the featuresMap
        spInfo=zeros(size(featuresVideo, 1), 2);
        k=1;
        for i=1:rows
            for j=1:cols
               spInfo(k*step-step+1:k*step, 1)=i;
               spInfo(k*step-step+1:k*step, 2)=j;
               k=k+1;             
            end
        end
        
        
    otherwise
       fprintf('Unkonwn parameter!!!! the layer should be fc8, fc7, fc6 conv5_3 ...'); 

end

