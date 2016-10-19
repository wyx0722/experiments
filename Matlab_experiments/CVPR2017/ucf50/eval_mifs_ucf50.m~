
path_features='/home/ionut/asustor_ionut/Data/MIFS/ucf50_l3/cache/'

dirFeatures=sprintf('%s*.mat',path_features);

list_mat_file=dir(dirFeatures);

cell_features=cell(1, length(list_mat_file));

fprintf('Load features from %d files!!  ... \n', length(list_mat_file));
for i=1:length(list_mat_file)
     fprintf('%d ', i);
     filename=[path_features list_mat_file(i).name];
     load(filename);
     cell_features{i}=data';    
end


all_data=cat(1, cell_features{i});