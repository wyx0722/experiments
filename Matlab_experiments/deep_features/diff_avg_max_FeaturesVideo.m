function  [avg_feat_vid, max_feat_vid]  = diff_avg_max_FeaturesVideo( pathVideo, layer )


switch layer
    
    case 'fc8_sVGG'
        dimFeatures=1000;
        fileID=fopen(pathVideo); %open the file
        [feat, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer

        %now each column represents the features for a frame within the
        %video, the next step is to compute the average pooling for each
        %line and then transpose the result to get a row vector for a video
        
        
        avg_feat_vid=mean(feat(:, 1:end-1)-feat(:, 2:end), 2)';
        max_feat_vid=max(feat(:, 1:end-1)-feat(:, 2:end),[], 2)';
        
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID); %close the opened file
   
            
    case 'fc8_tVGG_UCF101'
        dimFeatures=101;
        fileID=fopen(pathVideo); %open the file
        [feat, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer

        %now each column represents the features for a frame within the
        %video, the next step is to compute the average pooling for each
        %line and then transpose the result to get a row vector for a video
        avg_feat_vid=mean(feat(:, 1:end-1)-feat(:, 2:end), 2)';
        max_feat_vid=max(feat(:, 1:end-1)-feat(:, 2:end),[], 2)';
        
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID); %close the opened file
        
        
    case 'fc7'
        dimFeatures=4096;
        fileID=fopen(pathVideo); %open the file
        [feat, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer

        %now each column represents the features for a frame within the
        %video, the next step is to compute the average pooling for each
        %line and then transpose the result to get a row vector for a video
        avg_feat_vid=mean(feat(:, 1:end-1)-feat(:, 2:end), 2)';
        max_feat_vid=max(feat(:, 1:end-1)-feat(:, 2:end),[], 2)';
        
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID);  %close the opened file

    case 'fc6' 
        dimFeatures=4096;
        fileID=fopen(pathVideo); %open the file
        [feat, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer

        %now each column represents the features for a frame within the
        %video, the next step is to compute the average pooling for each
        %line and then transpose the result to get a row vector for a video
        avg_feat_vid=mean(feat(:, 1:end-1)-feat(:, 2:end), 2)';
        max_feat_vid=max(feat(:, 1:end-1)-feat(:, 2:end),[], 2)';
        
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID);  %close the opened file 
        
    case 'conv5_3' 
        dimFeatures=14*14*512;
        fileID=fopen(pathVideo); %open the file
        [feat, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer

        %now each column represents the features for a frame within the
        %video, the next step is to compute the average pooling for each
        %line and then transpose the result to get a row vector for a video
        avg_feat_vid=mean(feat(:, 1:end-1)-feat(:, 2:end), 2)';
        max_feat_vid=max(feat(:, 1:end-1)-feat(:, 2:end),[], 2)';
        
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID);  %close the opened file 
        
    case 'conv5_1' 
        dimFeatures=14*14*512;
        fileID=fopen(pathVideo); %open the file
        [feat, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer

        %now each column represents the features for a frame within the
        %video, the next step is to compute the average pooling for each
        %line and then transpose the result to get a row vector for a video
        avg_feat_vid=mean(feat(:, 1:end-1)-feat(:, 2:end), 2)';
        max_feat_vid=max(feat(:, 1:end-1)-feat(:, 2:end),[], 2)';
        
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID);  %close the opened file 
        
   case 'pool5' 
        dimFeatures=7*7*512;
        fileID=fopen(pathVideo); %open the file
        [feat, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer

        %now each column represents the features for a frame within the
        %video, the next step is to compute the average pooling for each
        %line and then transpose the result to get a row vector for a video
        avg_feat_vid=mean(feat(:, 1:end-1)-feat(:, 2:end), 2)';
        max_feat_vid=max(feat(:, 1:end-1)-feat(:, 2:end),[], 2)';
        
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID);  %close the opened file 
        
    case 'conv4_3' 
        dimFeatures=28*28*512;
        fileID=fopen(pathVideo); %open the file
        [feat, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer

        %now each column represents the features for a frame within the
        %video, the next step is to compute the average pooling for each
        %line and then transpose the result to get a row vector for a video
        avg_feat_vid=mean(feat(:, 1:end-1)-feat(:, 2:end), 2)';
        max_feat_vid=max(feat(:, 1:end-1)-feat(:, 2:end),[], 2)';
        
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:', mod(count, dimFeatures));
        end      
        fclose(fileID);  %close the opened file 
    
    otherwise
        fprintf('Unkonwn parameter!!!! the layer should be fc8, fc7, fc6 or conv4_3');
end



end

