function [ allFeatures ] = C3Dsave2mat( path_features, layer, savePath)
%ex:
%path_features='/home/ionut/asustor_ionut/work/del_c3d_fet/sarah_brushing_her_hair_brush_hair_h_cm_np1_ri_goo_1/'
%layer='pool5'
%savePath='/home/ionut/asustor_ionut/work/del_c3d_fet/sarah_brushing_her_hair_brush_hair_h_cm_np1_ri_goo_1_matF/'

dirFeatures=sprintf('%s*.%s',path_features, layer);

list_frames_feature=dir(dirFeatures);
allFeatures=cell(1, length(list_frames_feature));

for i=1:length(list_frames_feature)
    
    filename=[path_features list_frames_feature{1}.name];
    [s, blob, read_status] = read_binary_blob_preserve_shape(filename);
    
    if read_status~=1
        warning('The feature reading did not went well!!!!!\n%s', filename);
        keyboard
    end
    
    allFeatures{i}=blob;
end


fileNameSave=sprintf('%s%s.mat',savePath, layer);
save(fileNameSave, '-v7.3', allFeatures);
end

