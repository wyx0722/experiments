[vids, labs, groups] = GetVideosPlusLabels('Full');

load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/IDT/FEVid_IDTIDTfeatureHOFMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim54_Fisher_.mat');
idt_HOF=fisherV;

load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/IDT/FEVid_IDTIDTfeatureHOGMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48_Fisher_.mat');
idt_HOG=fisherV;

load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/IDT/FEVid_IDTIDTfeatureMBHxMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48_Fisher_.mat');
idt_MBHx=fisherV;

load('/home/ionut/Data/results/CBMI2015_rezults/videoRep/IDT/FEVid_IDTIDTfeatureMBHyMediaTypeIDTNormalisationROOTSIFTnumClusters256pcaDim48_Fisher_.mat');
idt_MBHy=fisherV;


load('/home/ionut//Data/results/CBMI2015_rezults/videoRep/frameSampleRate/FEVidHSMDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72_sRow3_Fisher_.mat');
hsm_fisher=fisherAll;


all_IDT01=cat(2, NormalizeRowsUnit(PowerNormalization(idt_HOF, 0.1)), NormalizeRowsUnit(PowerNormalization(idt_HOG, 0.1)), ...
                 NormalizeRowsUnit(PowerNormalization(idt_MBHx, 0.1)), NormalizeRowsUnit(PowerNormalization(idt_MBHy, 0.1)) );
        
all_IDT05=cat(2, NormalizeRowsUnit(PowerNormalization(idt_HOF, 0.5)),NormalizeRowsUnit(PowerNormalization(idt_HOG, 0.5)), ...
            NormalizeRowsUnit(PowerNormalization(idt_MBHx, 0.5)), NormalizeRowsUnit(PowerNormalization(idt_MBHy, 0.5)) );

hsm_fisher01=NormalizeRowsUnit(PowerNormalization(hsm_fisher, 0.1));


nEncoding=4;
allDist=cell(1, nEncoding);

ul_all_IDT01=NormalizeRowsUnit(all_IDT01);
allDist{1}=ul_all_IDT01 * ul_all_IDT01';
clear ul_all_IDT01


ul_all_IDT_hsm=NormalizeRowsUnit(cat(2,all_IDT01, hsm_fisher01 ));
allDist{2}=ul_all_IDT_hsm * ul_all_IDT_hsm';
clear ul_all_IDT_hsm


ul_all_IDT05=NormalizeRowsUnit(all_IDT05);
allDist{3}=ul_all_IDT05 * ul_all_IDT05';
clear ul_all_IDT05



allDist{4}=hsm_fisher01 * hsm_fisher01';


all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;

parpool(13);
for k=1:nEncoding

% 
% Leave-one-group-out cross-validation
parfor i=1:max(groups)
    testI = groups == i;
    trainI = ~testI;
    trainDist = allDist{k}(trainI, trainI);
    testDist = allDist{k}(testI, trainI);
    trainLabs = labs(trainI,:);
    testLabs = labs(testI, :);
    
    [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
    accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
    fprintf('%d: accuracy: %.3f\n', i, mean(accuracy{i}));
end

all_clfsOut{k}=clfsOut;
all_accuracy{k}=accuracy;

k
perGroupAccuracy = mean(cat(2, accuracy{:}))'

end


delete(gcp('nocreate'))


acc1=mean(mean(cat(2, all_accuracy{1}{:})))
acc2=mean(mean(cat(2, all_accuracy{2}{:})))
acc3=mean(mean(cat(2, all_accuracy{3}{:})))
acc4=mean(mean(cat(2, all_accuracy{4}{:})))






