function [desc, iTraj]= fast_descIDT(features, descType)



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

iTraj=0;

fileID=fopen(features);

if fileID==-1
    error('Could not open the file');
    keyboard;
end




switch descType
    
    case 'infoTraj'
        text=fscanf(fileID, '%f');
        %%%%%%%
        %!!!!!! Problem with some files that are possible corrupted, where not all the lines has 436 values.        
        if mod(size(text, 1),436)~=0
            warning('Possible corrupted file. mod(size(text, 1),436): %d \n file: %s', mod(size(text, 1),436), features);
            keepDesc=floor(size(text, 1)/436);
            text=text(1:keepDesc*436);    
        end
        %%%%%%%        
        
        
        desc=zeros(size(text, 1)/436, 10);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k:k+9);
              k=k+436;
        end
        
    
    case 'Traj'
        text=fscanf(fileID, '%f');
        %%%%%%%
        %!!!!!! Problem with some files that are possible corrupted, where not all the lines has 436 values.        
        if mod(size(text, 1),436)~=0
            warning('Possible corrupted file. mod(size(text, 1),436): %d \n file: %s', mod(size(text, 1),436), features);
            keepDesc=floor(size(text, 1)/436);
            text=text(1:keepDesc*436);    
        end
        %%%%%%%
        
        desc=zeros(size(text, 1)/436, 30);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k+10:k+39);
              k=k+436;
        end
        
        
    case 'HOG'

        text=fscanf(fileID, '%f');
        
        %%%%%%%
        %!!!!!! Problem with some files that are possible corrupted, where not all the lines has 436 values.        
        if mod(size(text, 1),436)~=0
            warning('Possible corrupted file. mod(size(text, 1),436): %d \n file: %s', mod(size(text, 1),436), features);
            keepDesc=floor(size(text, 1)/436);
            text=text(1:keepDesc*436);    
        end
        %%%%%%%
        
        
        desc=zeros(size(text, 1)/436, 96);
       
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k+40:k+135);
              k=k+436;
        end
    case 'HOF'
        text=fscanf(fileID, '%f');
        %%%%%%%
        %!!!!!! Problem with some files that are possible corrupted, where not all the lines has 436 values.        
        if mod(size(text, 1),436)~=0
            warning('Possible corrupted file. mod(size(text, 1),436): %d \n file: %s', mod(size(text, 1),436), features);
            keepDesc=floor(size(text, 1)/436);
            text=text(1:keepDesc*436);    
        end
        %%%%%%%
        
        desc=zeros(size(text, 1)/436, 108);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k+136:k+243);
              k=k+436;
        end
        
    case 'MBHx'

        text=fscanf(fileID, '%f');
        %%%%%%%
        %!!!!!! Problem with some files that are possible corrupted, where not all the lines has 436 values.        
        if mod(size(text, 1),436)~=0
            warning('Possible corrupted file. mod(size(text, 1),436): %d \n file: %s', mod(size(text, 1),436), features);
            keepDesc=floor(size(text, 1)/436);
            text=text(1:keepDesc*436);    
        end
        %%%%%%%
        
        desc=zeros(size(text, 1)/436, 96);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k+244:k+339);
              k=k+436;
        end
        
    case 'MBHy'

        text=fscanf(fileID, '%f');
        %%%%%%%
        %!!!!!! Problem with some files that are possible corrupted, where not all the lines has 436 values.        
        if mod(size(text, 1),436)~=0
            warning('Possible corrupted file. mod(size(text, 1),436): %d \n file: %s', mod(size(text, 1),436), features);
            keepDesc=floor(size(text, 1)/436);
            text=text(1:keepDesc*436);    
        end
        %%%%%%%
        
        desc=zeros(size(text, 1)/436, 96);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k+340:k+435);
              k=k+436;
        end
        
    case 'ALL'
        text=fscanf(fileID, '%f');
        %%%%%%%
        %!!!!!! Problem with some files that are possible corrupted, where not all the lines has 436 values.        
        if mod(size(text, 1),436)~=0
            warning('Possible corrupted file. mod(size(text, 1),436): %d \n file: %s', mod(size(text, 1),436), features);
            keepDesc=floor(size(text, 1)/436);
            text=text(1:keepDesc*436);    
        end
        %%%%%%%
        
        desc=zeros(size(text, 1)/436, 436);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k:k+435);
              k=k+436;
        end
        
        
        %%%%%%%%%%%%%for SP
        %%%%%%%%%%%%%%
        %%%%%%%%%%%%%%        
    case 'HOG_iTraj'

        text=fscanf(fileID, '%f');
        
        %%%%%%%
        %!!!!!! Problem with some files that are possible corrupted, where not all the lines has 436 values.        
        if mod(size(text, 1),436)~=0
            warning('Possible corrupted file. mod(size(text, 1),436): %d \n file: %s', mod(size(text, 1),436), features);
            keepDesc=floor(size(text, 1)/436);
            text=text(1:keepDesc*436);    
        end
        %%%%%%%
        
        
        desc=zeros(size(text, 1)/436, 96);
       
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k+40:k+135);
              k=k+436;
        end
        
        
        iTraj=zeros(size(text, 1)/436, 10);
        k=1;
        for i=1:size(iTraj, 1)
              iTraj(i, :)=text(k:k+9);
              k=k+436;
        end
        
    case 'HOF_iTraj'
        text=fscanf(fileID, '%f');
        %%%%%%%
        %!!!!!! Problem with some files that are possible corrupted, where not all the lines has 436 values.        
        if mod(size(text, 1),436)~=0
            warning('Possible corrupted file. mod(size(text, 1),436): %d \n file: %s', mod(size(text, 1),436), features);
            keepDesc=floor(size(text, 1)/436);
            text=text(1:keepDesc*436);    
        end
        %%%%%%%
        
        desc=zeros(size(text, 1)/436, 108);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k+136:k+243);
              k=k+436;
        end
        
        iTraj=zeros(size(text, 1)/436, 10);
        k=1;
        for i=1:size(iTraj, 1)
              iTraj(i, :)=text(k:k+9);
              k=k+436;
        end
        
    case 'MBHx_iTraj'

        text=fscanf(fileID, '%f');
        %%%%%%%
        %!!!!!! Problem with some files that are possible corrupted, where not all the lines has 436 values.        
        if mod(size(text, 1),436)~=0
            warning('Possible corrupted file. mod(size(text, 1),436): %d \n file: %s', mod(size(text, 1),436), features);
            keepDesc=floor(size(text, 1)/436);
            text=text(1:keepDesc*436);    
        end
        %%%%%%%
        
        desc=zeros(size(text, 1)/436, 96);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k+244:k+339);
              k=k+436;
        end
        
        iTraj=zeros(size(text, 1)/436, 10);
        k=1;
        for i=1:size(iTraj, 1)
              iTraj(i, :)=text(k:k+9);
              k=k+436;
        end
        
    case 'MBHy_iTraj'

        text=fscanf(fileID, '%f');
        %%%%%%%
        %!!!!!! Problem with some files that are possible corrupted, where not all the lines has 436 values.        
        if mod(size(text, 1),436)~=0
            warning('Possible corrupted file. mod(size(text, 1),436): %d \n file: %s', mod(size(text, 1),436), features);
            keepDesc=floor(size(text, 1)/436);
            text=text(1:keepDesc*436);    
        end
        %%%%%%%
        
        desc=zeros(size(text, 1)/436, 96);
        k=1;
        for i=1:size(desc, 1)
              desc(i, :)=text(k+340:k+435);
              k=k+436;
        end
        
        iTraj=zeros(size(text, 1)/436, 10);
        k=1;
        for i=1:size(iTraj, 1)
              iTraj(i, :)=text(k:k+9);
              k=k+436;
        end
        
        %%%%%%%%%%%%%
        %%%%%%%%%%%%%
        %%%%%%%%%%%%%
        
    otherwise
        warning('Unexpected choice!!!! Should be among:infoTraj, Traj, HOG, HOF, MBHx, MBXy, ALL');
end


fclose(fileID);

% command2=['rm ' filePath];
% [status2 cmdout2]=system(command2);
% 
% if status2~=0
%     warning('The System command did not proccess right: %s', command2);
%     %keyboard;
% end
