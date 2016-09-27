
layer{1}='pool4';
layer{2}='conv5b';
layer{3}='pool5';
layer{4}='fc6-1';
layer{5}='fc7-1';
layer{6}='prob';

allFeatures=cell(1, length(layer));

for i=1:length(layer)
     [ allFeatures{i} ] = C3Dsave2mat( path_features, layer(i), savePath);
end