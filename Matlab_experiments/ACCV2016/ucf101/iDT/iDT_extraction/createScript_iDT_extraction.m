videosList = textread('listAllVideos2.txt', '%s', 'delimiter', '\n');


fileID1=fopen('script_extracIDT_1.sh', 'w');
fileID2=fopen('script_extracIDT_2.sh', 'w');
fileID3=fopen('script_extracIDT_3.sh', 'w');
fileID4=fopen('script_extracIDT_4.sh', 'w');

string1='echo "video from list: $((i++))" \r\n';
baseDirSaveFeat='/home/ionut/asustor_ionut/Data/iDT_Features_UCF101/Videos/';

i1=1;
i2=1;
for i=1:length(videosList)
    
    if strfind(videosList{i}, '(')>0
        videosList{i}=strrep(videosList{i}, '(', '\\(');
        firstBra{i1}=i;
        i1=i1+1;
    end
    
    if strfind(videosList{i}, ')')>0
        videosList{i}=strrep(videosList{i}, ')', '\\)');
        secondtBra{i2}=i;
        i2=i2+1;
    end
    
    
    poz=strfind(videosList{i}, '/');
    videoName=videosList{i}(poz(1)+1:end);
    
    
    
    string2=['/home/ionut/IDT_code/improved_trajectory_release/release/DenseTrackStab /home/ionut/Data/UCF-101/Videos/' ...
            videosList{i} '.avi > ' baseDirSaveFeat videosList{i} '\r\n'];
    
    if i<=3330
        fprintf(fileID1, string1);
        fprintf(fileID1, string2);
    elseif i>3330 && i<=2*3330
        fprintf(fileID2, string1);
        fprintf(fileID2, string2);
    elseif i>2*3330 && i<=3*3330
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