function [  ] = create_in_out_C3D( videoList, pathFrames, pathSaveFeatures, inputFile, outputFile)
%create_in_out_C3D( videoList, pathFrames, pathSaveFeatures, inputFile, outputFile)
% input:
% videoList: the list of videos
% pathFrames: the base path where the frames are saved
% pathSaveFeatures: the base bath where the features will be saved
% inputFile: the desired name for the input file that will be created
% outputFile: the desired name for the output file that will be created

fileID=fopen(videoList);
listV=textscan(fileID, '%s');
fclose(fileID);

fileID_in=fopen(inputFile, 'w');
fileID_out=fopen(outputFile, 'w');

for i=1:length(listV{1})
    
    dirFrames=sprintf('%s%s/*.jpg',pathFrames, listV{1}{i})
    list_frames=dir(dirFrames);
    for j=1:(length(list_frames) - 15) %!!!!       
    fprintf(fileID_in, '%s%s/ %d 0\n',pathFrames, listV{1}{i}, j);
    fprintf(fileID_out, '%s%s/%s\n',pathSaveFeatures,listV{1}{i}, list_frames(j).name(1:end-4));
    end
    
end

fclose(fileID_in);
fclose(fileID_out);


end

