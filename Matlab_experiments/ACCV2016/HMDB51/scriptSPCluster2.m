
spl{1}='HMBD51Split1';
spl{2}='HMBD51Split2';
spl{3}='HMBD51Split3';

%for s=2:3
    
    
% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='SPVGG19'; %!!!!!
descParam.Normalisation='None';


descParam.pcaDim = 128;

descParam.orgClusters=256;

descParam.smallCL=32;
Cl=[8 32 64];


bazePathFeatures='/media/HDS2-UTX/ionut/Data/hmdb51_VGG_19_features_rawFrames/Videos/'


for i=1:length(spl)
    descParam.Dataset=spl{i};%descParam.Dataset=spl{s};
    
    for j=1:length(Cl)
    
        descParam.bovwCL=Cl(j);
        descParam
        
        [all_accuracy, all_clfsOut ]=hmdb51Framework(descParam, bazePathFeatures );
        
        acc1=mean(all_accuracy{1})
        acc2=mean(all_accuracy{2})
        acc3=mean(all_accuracy{3})
        acc4=mean(all_accuracy{4})
        acc5=mean(all_accuracy{5})
        acc6=mean(all_accuracy{6})
        acc7=mean(all_accuracy{7})
        acc8=mean(all_accuracy{8})
        
        
        fileName=sprintf('/home/ionut/experiments/Matlab_experiments/ACCV2016/results/results_HMDB51_Features%s_Layer%s_Network%s_PCAdim%d_norm%s.txt', ...
                descParam.MediaType,descParam.Layer, descParam.net,descParam.pcaDim, descParam.Normalisation);
            
        fileID=fopen(fileName, 'a');

        fprintf(fileID, 'Dataset and Split: %s, orgClusters:%d, bovwCL:%d, smallCL%d, with intraL2 and at the end PNL2 --> \n --> vladNoMean acc= %.4f   SD acc= %.4f  M-VLAD acc= %.4f  M-SD acc= %.4f  SD-VLAD acc= %.4f  SDM-VLAD (M-VLAD + M-SD) acc= %.4f   VLAD + M-VLAD acc= %.4f  final SDM-VLAD (VLAD + SD + M-SD + SDM-VLAD) acc= %.4f \r\n\n', ...
             descParam.Dataset,descParam.orgClusters, descParam.bovwCL, descParam.smallCL, acc1,acc2, acc3, acc4, acc5, acc6, acc7,acc8 );
        fclose(fileID);
   end
end