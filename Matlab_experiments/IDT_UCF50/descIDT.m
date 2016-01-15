function [desc]= descIDT(features, descType)



pozF=strfind(features, '.');
filePath=features(1:pozF(end)-1);

command=['gunzip -c ' features ' > ' filePath];


[status cmdout]=system(command);

if status~=0
    error('The System command did not complete right: %s', command);
    keyboard;
end



fileID=fopen(filePath);

if fileID==-1
    error('Could not open the file');
    keyboard;
end




switch descType
    
    case 'infoTraj'
        line=fgetl(fileID);
        desc=[];
        while ischar(line)
            vLine=str2num(line);
            desc=cat(1, desc, vLine(1:10));
            line=fgetl(fileID);
        end
    
    case 'Traj'
        line=fgetl(fileID);
        desc=[];
        while ischar(line)
            vLine=str2num(line);
            desc=cat(1, desc, vLine(11:40));
            line=fgetl(fileID);
        end
        
    case 'HOG'
        line=fgetl(fileID);
        desc=[];
        while ischar(line)
            vLine=str2num(line);
            desc=cat(1, desc, vLine(41:136));
            line=fgetl(fileID);
        end
        
    case 'HOF'
        line=fgetl(fileID);
        desc=[];
        while ischar(line)
            vLine=str2num(line);
            desc=cat(1, desc, vLine(137:244));
            line=fgetl(fileID);
        end
        
    case 'MBHx'
        line=fgetl(fileID);
        desc=[];
        while ischar(line)
            vLine=str2num(line);
            desc=cat(1, desc, vLine(245:340));
            line=fgetl(fileID);
        end
        
    case 'MBHy'
        line=fgetl(fileID);
        desc=[];
        while ischar(line)
            vLine=str2num(line);
            desc=cat(1, desc, vLine(341:436));
            line=fgetl(fileID);
        end
        
    case 'ALL'
        line=fgetl(fileID);
        desc=[];
        while ischar(line)
            vLine=str2num(line);
            desc=cat(1, desc, vLine);
            line=fgetl(fileID);
        end
        
    otherwise
        warning('Unexpected choice!!!! Should be among:infoTraj, Traj, HOG, HOF, MBHx, MBXy, ALL');
end


fclose(fileID);

command2=['rm ' filePath];
[status2 cmdout2]=system(command2);

if status2~=0
    warning('The System command did not proccess right: %s', command2);
    keyboard;
end
