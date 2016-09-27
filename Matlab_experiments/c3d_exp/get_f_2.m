function [ allFeatures ] = get_f_2( pathFeatures, layer )


if ~isempty(strfind(layer, 'pool5'))
    rows=4;
    cols=4;  
    channels=512;
end


loadName=sprintf('%s%s.mat',pathFeatures, layer);

load(loadName)


allFeatures=zeros(rows*cols*length(allFeatures), channels);
p=1;

for i=1:length(allFeatures)
    
    
    for m=1:size(allFeatures{i}, 4)
        for n=size(allFeatures{i}, 5)
           allFeatures(p, :)=allFeatures{i}(1, :, 1, m, n);
           p=p+1;
        end
    end
end


end

