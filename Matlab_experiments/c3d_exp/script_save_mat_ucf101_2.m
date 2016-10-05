
%brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0

fileID=fopen('/home/ionut/Data/UCF-101/listAllVideos2.txt');
listV=textscan(fileID, '%s');
fclose(fileID);

base_frames_features='/srv/datasets/UCF101_c3d/Videos/'
base_save_mat='/home/ionut/Data/mat_c3d_features_ucf101/Videos/'

layer{1}='pool4';
layer{2}='conv5b';
layer{3}='pool5';
layer{4}='fc6-1';
layer{5}='fc7-1';
layer{6}='prob';


for i=3330:6660 %i=1:length(listV{1})
    
    path_features=sprintf('%s%s/', base_frames_features, listV{1}{i});
    savePath=sprintf('%s%s/', base_save_mat, listV{1}{i});
    fprintf('%d -> %s && %s\n', i, path_features,  savePath);
    
   for j=1:length(layer)
     C3Dsave2mat( path_features, layer{j}, savePath);
   end
         
end