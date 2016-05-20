videosList = textread('listVideos.txt', '%s', 'delimiter', '\n');


fileID1=fopen('script_extracIDT_1.sh', 'w');
fileID2=fopen('script_extracIDT_2.sh', 'w');
fileID3=fopen('script_extracIDT_3.sh', 'w');
fileID4=fopen('script_extracIDT_4.sh', 'w');

string1='echo "video from list: $((i++))" \r\n';
baseDirSaveFeat='/home/ionut/Data/iDT_Features_HMDB51/Videos/';

for i=1:length(videosList)
    
    if strfind(videosList{i}, '(')>0
        videosList{i}=strrep(videosList{i}, '(', '\\(');
        videosList{i}=strrep(videosList{i}, ')', '\\)');
    end
    poz=strfind(videosList{i}, '/');
    videoName=videosList{i}(poz(1)+1:end);
    
    
    
    string2=['/home/ionut/IDT_code/improved_trajectory_release/release/DenseTrackStab /home/ionut/Data/HMDB51/Videos/' ...
            videosList{i} '.avi -H /home/ionut/IDT_code/bb_file/HMDB51/' videoName '.bb > ' baseDirSaveFeat videosList{i} '\r\n'];
    
    if i<=1691
        fprintf(fileID1, string1);
        fprintf(fileID1, string2);
    elseif i>1691 && i<=2*1691
        fprintf(fileID2, string1);
        fprintf(fileID2, string2);
    elseif i>2*1691 && i<=3*1691
        fprintf(fileID3, string1);
        fprintf(fileID3, string2); 
    else
        fprintf(fileID4, string1);
        fprintf(fileID4, string2);
        
    end
end

fclose(fileID1);
fclose(fileID2);
fclose(fileID3);
fclose(fileID4);