function [  ] = unzipFile_DeleteGz( file )

pozF=strfind(file, '.');
filePath=file(1:pozF(end)-1);

command=['gunzip -c ' file ' > ' filePath];


[status cmdout]=system(command);

if status~=0
    error('The System command did not complete right: %s \n file: %s', command, file);
    keyboard;
end


command2=['rm ' file];
[status2 cmdout2]=system(command2);

if status2~=0
    warning('The System command did not proccess right: %s \n file: %s', command2, file);
    keyboard;
end


