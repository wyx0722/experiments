[vids, labs, groups] = GetVideosPlusLabels('Full');


bazePathFeatures='/home/ionut/Features/Features/UCF50/IDT/Videos/'; %change


pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i)) '.gz'];
end

parpool(10);
parfor i=1:length(pathFeatures)
    
    fprintf('%d \n', i)
    unzipFile_DeleteGz( pathFeatures{i} );
    
end


delete(gcp('nocreate'))