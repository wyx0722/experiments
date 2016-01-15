function [desc, infoTraj]= stHOG_fast_descIDT2(features, descType)



% pozF=strfind(features, '.');
% filePath=features(1:pozF(end)-1);
% 
% command=['gunzip -c ' features ' > ' filePath];
% 
% 
% [status cmdout]=system(command);
% 
% if status~=0
%     error('The System command did not complete right: %s', command);
%     keyboard;
% end



%fileID=fopen(filePath);
fileID=fopen(features); %!!!!!!!!!!!delete

if fileID==-1
    error('Could not open the file');
    keyboard;
end


if nargin<2
    descType='HOG_infoTraj';  
end


switch descType
    
    case 'infoTraj'
        text=fscanf(fileID, '%f');
        desc=zeros(size(text, 1)/106, 10);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k:k+9);
              k=k+106;
        end
        
 
    case 'HOG'

        text=fscanf(fileID, '%f');
        desc=zeros(size(text, 1)/106, 96);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k+10:k+105);
              k=k+106;
        end
    
        
    case 'ALL'
        text=fscanf(fileID, '%f');
        desc=zeros(size(text, 1)/106, 106);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k:k+105);
              k=k+106;
        end
        
    case 'HOG_infoTraj'
        
        text=fscanf(fileID, '%f');
        desc=zeros(size(text, 1)/106, 96);
        infoTraj=zeros(size(text, 1)/106, 10);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k+10:k+105);
              infoTraj(i, :)=text(k:k+9);
              k=k+106;
        end
        
    otherwise
        warning('Unexpected choice!!!! Should be among:infoTraj, Traj, HOG, ALL');
end


fclose(fileID);

% command2=['rm ' filePath];
% [status2 cmdout2]=system(command2);
% 
% if status2~=0
%     warning('The System command did not proccess right: %s', command2);
%     keyboard;
% end
