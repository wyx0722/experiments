videosList = textread('/Users/Ionut/Disk_E/experiments/Matlab_experiments/video-list.txt', '%s', 'delimiter', '\n');


fileID=fopen('script_evalEff_IDT.sh', 'w');
string1='echo "variable inc: $((i++))" \r\n';

for i=1:length(videosList)
    
    fprintf(fileID, string1);
    string2=['/home/ionut/IDT_code/improved_trajectory_release/release/DenseTrackStab /home/ionut/Data/UCF50/Videos/' ...
            videosList{i} '.avi | gzip > /home/ionut/IDT_code/improved_trajectory_release/del_out.features.gz \r\n'];
    
    fprintf(fileID, string2);
    
end

fclose(fileID);
