function [ featuresVideo, spInfo ] = maps2features( pathFeaturesVideo, layer, norm )

%pathFeaturesVideo - represents the file where the deep features are saved

%each row in the file represents the features for a frame, so the number of
%elemts for each row is: the dimensionality of the Conv. maps times the
%number of channels

%the values are saved in the file in the index order of the array: (k(i(j))),
%where k is the number of channels, i is the number of rows of ConvMaps and
%j is the number of columns oc ConvMaps, so the first (i x j) elements of a row
%represents all the values of the first channel etc ...


%featuresVideo=[];
%spInfo=[];

switch layer
    
    case {'pool4', 'conv5_1', 'conv5_3', 'conv5_4'}
        rows=14;
        cols=14;  
        channels=512;
        getFeatures
        
    case 'pool5'
        rows=7;
        cols=7;
        channels=512;
        getFeatures
        
    case {'fc6', 'fc7'}
        rows=1;
        cols=1;
        channels=4096;
        
        dimConvMap=rows*cols;
        dimFeatures=dimConvMap * channels;
        fileID=fopen(pathFeaturesVideo); %open the file
        [featuresVideo, count]=fscanf(fileID, '%f', [dimFeatures inf]); %read the opened file with the format coresponding to the layer
        featuresVideo=featuresVideo';
        nFeatursMaps=size(featuresVideo,1);
        if mod(count, dimFeatures)~=0 %supplementary check if there is something wrong with the dimension
            fprintf('warning!!!!, check feature dimension, mod:  file: %s', mod(count, dimFeatures), pathFeaturesVideo);
        end      
        fclose(fileID);
        
        spInfo=ones(size(featuresVideo, 1), 2);
        
    otherwise
       fprintf('Unkonwn parameter!!!! the layer should be pool4, conv5_1, conv5_3, conv5_4, pool5, fc6, fc7   ...'); 
end
         
          
function getFeatures        

    dimConvMap=rows*cols;
    dimFeatures=dimConvMap * channels;
    fileID=fopen(pathFeaturesVideo); %open the file
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

    if nargin>2
    switch norm
        case 'stNorm' %spatiotemporal normalization. Normalize each channel independentely to its maximum value 
            maxST=max(featuresVideo, [], 1);
            featuresVideo=bsxfun(@rdivide, featuresVideo, maxST);
            featuresVideo(isnan(featuresVideo))=0;
        case 'chNorm' %channel normalization. Normalize each feature acording to its location
            step=nFeatursMaps;
            for i=1:dimConvMap
                featuresVideo(i*step-step+1:i*step,:)=featuresVideo(i*step-step+1:i*step,:) ./ max(max(featuresVideo(i*step-step+1:i*step,:))); 
            end
            featuresVideo(isnan(featuresVideo))=0;

        case 'st_ch_Norm'
            maxST=max(featuresVideo, [], 1);
            featuresVideo=bsxfun(@rdivide, featuresVideo, maxST);
            featuresVideo(isnan(featuresVideo))=0;

            step=nFeatursMaps;
            for i=1:dimConvMap
                featuresVideo(i*step-step+1:i*step,:)=featuresVideo(i*step-step+1:i*step,:) ./ max(max(featuresVideo(i*step-step+1:i*step,:))); 
            end
            featuresVideo(isnan(featuresVideo))=0;

        case  'ch_st_Norm'
            step=nFeatursMaps;
            for i=1:dimConvMap
                featuresVideo(i*step-step+1:i*step,:)=featuresVideo(i*step-step+1:i*step,:) ./ max(max(featuresVideo(i*step-step+1:i*step,:))); 
            end
            featuresVideo(isnan(featuresVideo))=0;

            maxST=max(featuresVideo, [], 1);
            featuresVideo=bsxfun(@rdivide, featuresVideo, maxST);
            featuresVideo(isnan(featuresVideo))=0;

        case 'None'
            %no normalization of features maps

        otherwise
            error('Unknown parameter for feture normalization. Shuld be  stNorm, chNorm, st_ch_Norm, ch_st_Norm')

    end
    end
end

end

