videosList = textread('listVideos.txt', '%s', 'delimiter', '\n');


fileID1=fopen('fixError_script_extracIDT.sh', 'w');

string1='echo "video from list: $((i++))" \r\n';
baseDirSaveFeat='/home/ionut/Data/iDT_Features_HMDB51/Videos/';

i1=1;
i2=1;
i3=1;
i4=0;
sc=0;
a_nd=0;
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
    
    if strfind(videosList{i}, ';')>0
        videosList{i}=strrep(videosList{i}, ';', '\\;');
        semiColumn{i3}=i;
        i3=i3+1;
        sc=1;
    end
    
        if strfind(videosList{i}, '&')>0
        videosList{i}=strrep(videosList{i}, '&', '\\&');
        aa_nd{i3}=i;
        i3=i3+1;
        a_nd=1;
    end
    
    poz=strfind(videosList{i}, '/');
    videoName=videosList{i}(poz(1)+1:end);
    
    
    
    string2=['/home/ionut/IDT_code/improved_trajectory_release/release/DenseTrackStab /home/ionut/Data/HMDB51/Videos/' ...
            videosList{i} '.avi -H /home/ionut/IDT_code/bb_file/HMDB51/' videoName '.bb > ' baseDirSaveFeat videosList{i} '\r\n'];
    
    if sc==1
        fprintf(fileID1, string1);
        fprintf(fileID1, string2);
        sc=0;
    elseif a_nd==1
        fprintf(fileID1, string1);
        fprintf(fileID1, string2);
        a_nd=0; 
        
        
    end
    
end

fclose(fileID1);
