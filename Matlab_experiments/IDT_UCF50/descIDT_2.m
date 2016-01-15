function [desc]= descIDT(features, descType)


tic;

poz=strfind(features, '/');
fPath=features(1:poz(end)-1);

command=['tar -C ' fPath ' -zxvf ' features];

[status cmdout]=system(command);

if status~=0
    error('The System command did not proccess right: %s', command);
    keyboard;
end


pozF=strfind(features, '.');
filePath=features(1:pozF(end)-1);


fileID=fopen(filePath);

if fileID==-1
    error('Could not open the file');
    keyboard;
end

toc

tic
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
            %vLine=str2num(line);
            %desc=cat(1, desc, vLine(11:40));
           desc=cat(1, desc, line(11:40));
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

toc

tic
fclose(fileID);


command2=['rm ' filePath];
[status2 cmdout2]=system(command2);

if status2~=0
    warning('The System command did not proccess right: %s', command2);
    keyboard;
end

toc
% tic
% fileID=fopen('person01_boxing_d1')
% 
% if fileID==-1
%     error('Could not open the file');
%     keyboard;
% end
% 
% line = fgetl(fileID);
% hogs=[];
% while ischar(line)
%     vLine=str2num(line);
%     hogs=cat(1, hogs, vLine(41:136));
%     line=fgetl(fileID);
% end
% fclose(fileID);
% 
% toc
% 
% tic
% command2='rm person01_boxing_d1';
% status2=system(command2)
% toc
% 



