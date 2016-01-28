

v1_meanAcc=mean(mean(cat(2, all_accuracy{1}{:})));
v2_meanAcc=mean(mean(cat(2, all_accuracy{2}{:})));
v3_meanAcc=mean(mean(cat(2, all_accuracy{3}{:})));

try
    
    
    fileName=['/home/ionut/Data/results_desc_IDT/rez_newNorm/results_UCF50_newNorm__' 'Desc' descParam.MediaType '_' descParam.IDTfeature '_norm' descParam.Normalisation '.txt'];
    
    fileID=fopen(fileName, 'a');
    
    fprintf(fileID, '%s %s norm:%s  alpha: %.2f --> v1: %.3f  v2: %.3f  v3: %.3f \r\n', ...
            descParam.MediaType,descParam.IDTfeature, descParam.Normalisation,descParam.alpha,v1_meanAcc,v2_meanAcc, v3_meanAcc);
    
    fclose(fileID);
    

    


            
    
catch err
     fileName=['/home/ionut/Data/results_desc_IDT/rez_newNorm/backup/backup_results_UCF50_newNorm__' 'Desc' descParam.MediaType '_' descParam.IDTfeature '_norm' descParam.Normalisation '.txt'];
    
    fileID=fopen(fileName, 'a');
    
    fprintf(fileID, '%s %s norm:%s  alpha: %.2f --> v1: %.3f  v2: %.3f  v3: %.3f \r\n', ....
              descParam.MediaType,descParam.IDTfeature, descParam.Normalisation,descParam.alpha,v1_meanAcc,v2_meanAcc, v3_meanAcc);
    fclose(fileID);
    
    warning('error writing %s. Instead the file%s was saved in: ',err, fileName);
        
end

