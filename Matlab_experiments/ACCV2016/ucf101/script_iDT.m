
    
% Parameter settings for descriptor extraction
descParam.Func = @FEVid_IDT;
descParam.MediaType = 'IDT';
%descParam.IDTfeature='HOF';
descParam.Normalisation='ROOTSIFT'; % L2 or 'ROOTSIFT'
descParam.Dataset='UCF101';


if strfind(descParam.IDTfeature, 'HOF')>0
    descParam.pcaDim = 56;
else
    descParam.pcaDim = 48;
end

descParam.orgClusters=256;
descParam.bovwCL=16;
descParam.smallCL=32;


bazePathFeatures='/home/ionut/asustor_ionut/Data/iDT_Features_UCF101/Videos/'

nPar=10;

for i=1:3
    descParam.Split=i;%descParam.Dataset=spl{s};
    

    descParam

    [all_accuracy, all_clfsOut ]=ucf101Framework(descParam, bazePathFeatures, nPar );

    acc1=mean(all_accuracy{1})
    acc2=mean(all_accuracy{2})
    acc3=mean(all_accuracy{3})
    acc4=mean(all_accuracy{4})
    acc5=mean(all_accuracy{5})
    acc6=mean(all_accuracy{6})
    acc7=mean(all_accuracy{7})
    acc8=mean(all_accuracy{8})
    acc9=mean(all_accuracy{9})

    fileName=sprintf('/home/ionut/experiments/Matlab_experiments/ACCV2016/results/ucf101/results_ucf101_Features%s_Desc%s_PCAdim%d_norm%s.txt', ...
            descParam.MediaType, descParam.IDTfeature, descParam.pcaDim, descParam.Normalisation);

    fileID=fopen(fileName, 'a');

    fprintf(fileID, 'Dataset and Split: %s:%d, orgClusters:%d, bovwCL:%d, smallCL%d, with intraL2 and at the end PNL2 --> \n --> vladNoMean acc= %.4f   SD acc= %.4f  M-VLAD acc= %.4f  M-SD acc= %.4f  SD-VLAD acc= %.4f  SDM-VLAD (M-VLAD + M-SD) acc= %.4f   VLAD + M-VLAD acc= %.4f  final SDM-VLAD (VLAD + SD + M-SD + SDM-VLAD) acc= %.4f noMeanVLAD512 acc= %.4f \r\n\n', ...
         descParam.Dataset,descParam.Split, descParam.orgClusters, descParam.bovwCL, descParam.smallCL, acc1,acc2, acc3, acc4, acc5, acc6, acc7,acc8, acc9 );
    fclose(fileID);
   
end