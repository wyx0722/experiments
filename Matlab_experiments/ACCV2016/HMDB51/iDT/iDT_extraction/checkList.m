videosList = textread('listVideos.txt', '%s', 'delimiter', '\n');

notComplete_videosList = textread('notCompleteList', '%s', 'delimiter', '\n');
k=1;
notInList=[];
for i=1:length(videosList)
    found=0;
    for j=1:length(notComplete_videosList)
        if strfind(notComplete_videosList{j}, videosList{i})>0
        found=1;
        break
        end
    end
    
    if found==0
        notInList{k}=videosList{i};
        k=k+1;
    end
end