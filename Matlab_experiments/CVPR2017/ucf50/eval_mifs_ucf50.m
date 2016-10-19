% 
% path_features='/home/ionut/asustor_ionut/Data/MIFS/ucf50_l3/'
% 
% dirFeatures=sprintf('%scache/*.mat',path_features);
% 
% list_mat_file=dir(dirFeatures);
% 
% cell_features=cell(1, length(list_mat_file));
% 
% fprintf('Load features from %d files!!  ... \n', length(list_mat_file));
% for i=1:length(list_mat_file)
%      fprintf('%d ', i);
%      filename=[path_features list_mat_file(i).name];
%      load(filename);
%      cell_features{i}=data';    
% end
% 
% 
% all_data=cat(1, cell_features{:});


load([ path_features 'imdb.mat']);



global DATAopts;
DATAopts = UCFInit;


[allVids, labs, groups] = GetVideosPlusLabels('Full');

representation=zeros(length(allVids), size(all_data, 2));

for i=1:length(allVids)
    if mod(i, 100)==0
        fprintf('%d ', i);
    end
    
    for j=1:(length(images.name))
        
        if ~isempty(strfind([meta.classes{images.class(j)} '/' images.name{j}(1:end-4)], allVids{i}))
            representation(i, :)=all_data(j, :);
            continue
        end
        
    end 
    
    
end
