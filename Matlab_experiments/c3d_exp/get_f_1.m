function [ allFeatures ] = get_f_1( pathFeatures, layer )


if ~isempty(strfind(layer, 'pool4'))
    rows=4;
    cols=4;  
    channels=512;
end


dirFeatures=sprintf('%s*.%s',pathFeatures, layer);

list_frames_feature=dir(dirFeatures);

allFeatures=zeros(rows*cols*length(list_frames_feature), channels);
p=1;

for i=1:length(list_frames_feature)
    
    filename=[path_features list_frames_feature(i).name];
    [s, blob, read_status] = read_binary_blob_preserve_shape(filename);
    
    if read_status~=1
        warning('The feature reading did not went well!!!!!\n%s', filename);
        keyboard
    end
    
    for m=1:size(blob, 4)
        for n=size(blob, 5)
           allFeatures(p, :)=blob(1, :, 1, m, n);
           p=p+1;
        end
    end
end


end

