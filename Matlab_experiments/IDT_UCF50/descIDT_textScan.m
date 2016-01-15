function [desc]= descIDT_textScan(features, descType)



pozF=strfind(features, '.');
filePath=features(1:pozF(end)-1);

% command=['gunzip -c ' features ' > ' filePath];
% 
% 
% [status cmdout]=system(command);
% 
% if status~=0
%     error('The System command did not complete right: %s', command);
%     keyboard;
% end



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
      tic  
    case 'HOG'
        line=fgetl(fileID);
        desc=[];
        while ischar(line)
            vLine=str2num(line);
            desc=cat(1, desc, vLine(41:136));
            line=fgetl(fileID);
        end
        toc
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
        tic
    case 'ALL'
        line=fgetl(fileID);
        desc=[];
        while ischar(line)
            vLine=str2num(line);
            desc=cat(1, desc, vLine);
            line=fgetl(fileID);
        end
        toc
        
        tic
        
    case 'ALL2'
        %line=fgetl(fileID);
      text=textscan(fileID, '%f', 'delimiter', '\n');
      
      desc=zeros(size(text{1}, 1)/436, 436);
      
      k=1;
      for i=1:size(desc, 1)
          desc(i, :)=text{1}(k:k+435);   %hog  desc(i, :)=text{1}(K+40:k+135)
          k=k+436;
      end
        toc
        
        
        case 'ALL3'
        %line=fgetl(fileID);
      text=fscanf(fileID, '%f');
      
      desc=zeros(size(text, 1)/436, 436);
      
      k=1;
      for i=1:size(desc, 1)
          desc(i, :)=text(k:k+435);   %hog  desc(i, :)=text{1}(K+40:k+135)
          k=k+436;
      end
        toc
        
        
        tic
        
    case 'HOG2'
        
      text=textscan(fileID, '%f', 'delimiter', '\n');
      
      desc=zeros(size(text{1}, 1)/436, 96);
      k=1;
      for i=1:size(desc, 1)
          desc(i, :)=text{1}(k+40:k+135);   %hog  desc(i, :)=text{1}(K+40:k+135)
          k=k+436;
      end
        
        
        toc
    otherwise
        warning('Unexpected choice!!!! Should be among:infoTraj, Traj, HOG, HOF, MBHx, MBXy, ALL');
end


fclose(fileID);

% command2=['rm ' filePath];
% [status2 cmdout2]=system(command2);
% 
% if status2~=0
%     warning('The System command did not proccess right: %s', command2);
%     keyboard;
% end
