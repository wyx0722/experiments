addpath('./../');%!!!!!!!
addpath('./../../');%!!!!!!!


global DATAopts;
DATAopts = UCFInit;

datasetName='UCF50';
mType='DeepF';
typeFeature=@FEVid_deepFeatures;
normStrategy='None';

%~~~~~~~~~~~~~~
layer='pool5'
Net='tempVGG16';
allPathFeatures='/home/ionut/asustor_ionut/Data/action_temporal_vgg_16_split1_features_opticalFlow_tvL1_UCF50/Videos/'%%%%%channge~~~~~~~~~~~~
%~~~~~~~~~~~~~~


d=[256 0];

alpha=0.5;
nPar=5;



descParam.Dataset=datasetName;
descParam.MediaType=mType;
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;

descParam.Clusters=[256];
descParam.spClusters=[32 64];
descParam.gmmSize=[256];


switch descParam.MediaType
    
    case 'Vid'
        descParam.NumBlocks = [3 3 2];
    
        if exist('fsr', 'var')
            descParam.FrameSampleRate = fsr;
            descParam.BlockSize = [8 8 6/fsr];
        else
            descParam.BlockSize = [8 8 6];
        end

        descParam.NumOr = 8;
        sRow = [1 3];
        sCol = [1 1];
        
        sizeDesc=144;
        if exist('d', 'var')
            descParam.pcaDim=d;
        else
            descParam.pcaDim=sizeDesc/2;
        end
        
    case 'IDT'
        descParam.IDTfeature=iDTfeature;
        
        if ~isempty(strfind(descParam.IDTfeature,'HOF'))
            sizeDesc=108;   
        else%if  strfind(descParam.IDTfeature,'HOG')>0 || strfind(descParam.IDTfeature,'MBHx')>0 || strfind(descParam.IDTfeature,'MBHy')>0  
            sizeDesc=96;   
        end
        if exist('d', 'var')
            descParam.pcaDim=d;
        else
            descParam.pcaDim=sizeDesc/2;
        end
        bazePathFeatures=allPathFeatures
        
    case 'DeepF'
        descParam.Layer=layer;
        descParam.net=Net;
 
        sizeDesc=512;
        if exist('d', 'var')
            descParam.pcaDim=d;
        else
            descParam.pcaDim=sizeDesc/2;
        end
        bazePathFeatures=allPathFeatures
end


descParam







vocabularyIms = GetVideosPlusLabels('smallEnd');
vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    if ~isempty(strfind(descParam.MediaType, 'DeepF'))
        if ~isempty(strfind(descParam.net, 'C3D'))
            file_extension='.mat';
        else
            file_extension='.txt';
        end
        vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '/' descParam.Layer file_extension];
    elseif strfind(descParam.MediaType, 'IDT')>0 
        vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i))];
    end
end


%vocabularyPathFeatures=allPathFeatures(1:4:end);%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[cell_Clusters, cell_spClusters, cell_pcaMap, cell_gmmModelName] = CreateVocabularyKmeansPca_sptCl_multiplePCA_KmeansAndFV(vocabularyImsPaths, descParam);

%the size of cell_spClusters is length(descParam.pcaDim)*length(descParam.Clusters)!!
%the size of cell_spClusters is 1*length(descParam.spClusters)
%the size 0f cell_pcaMap is 1*length(descParam.pcaDim)
%the size of cell_gmmModelName is length(descParam.pcaDim)*length(descParam.gmmSize)!!

% Now create set
[allVids, labs, groups] = GetVideosPlusLabels('Full');
allPathFeatures=cell(size(allVids));

for i=1:length(allPathFeatures)
    if ~isempty(strfind(descParam.MediaType, 'DeepF')) 
         if ~isempty(strfind(descParam.net, 'C3D'))
            file_extension='.mat';
        else
            file_extension='.txt';
        end
        allPathFeatures{i}=[bazePathFeatures char(allVids(i)) '/' descParam.Layer file_extension];
    elseif strfind(descParam.MediaType, 'IDT')>0  
        allPathFeatures{i}=[bazePathFeatures char(allVids(i))];
    end
end



[desc, info, descParam] = descParam.Func(allPathFeatures{1}, descParam);


pca256_desc = desc * cell_pcaMap{1}.data.rot;
pca0_desc = desc * cell_pcaMap{2}.data.rot;




t=ST_VLMPF_abs(pca256_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); sp32cl256pca256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca0_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); sp32cl256pca0=zeros(length(allPathFeatures), length(t), 'like', t);

t=ST_VLMPF_abs(pca256_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); sp64cl256pca256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(pca0_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); sp64cl256pca0=zeros(length(allPathFeatures), length(t), 'like', t);


t=VLAD_1(pca256_desc, cell_Clusters{1}.vocabulary); v256pca256=zeros(length(allPathFeatures), length(t), 'like', t);
t=VLAD_1(pca0_desc, cell_Clusters{2}.vocabulary); v256pca0=zeros(length(allPathFeatures), length(t), 'like', t);

t=mexFisherAssign(pca256_desc', cell_gmmModelName{1})'; fv256pca256=zeros(length(allPathFeatures), length(t), 'like', t);
t=mexFisherAssign(pca0_desc', cell_gmmModelName{2})'; fv256pca0=zeros(length(allPathFeatures), length(t), 'like', t);


clear pca256_desc pca0_desc desc info


nDesc=zeros(1, length(allPathFeatures));

parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(allPathFeatures));
parfor i=1:length(allPathFeatures)%parfor i=1:length(allPathFeatures)
%     if mod(i, 100)==0
%         fprintf('%d ', i)%fprintf('%d \n', i)
%     end
    
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = descParam.Func(allPathFeatures{i}, descParam);
    nDesc(i)=size(desc,1);
    

    pca256_desc = desc * cell_pcaMap{1}.data.rot;
    pca0_desc = desc * cell_pcaMap{2}.data.rot;

 
    
    sp32cl256pca256(i, :)=ST_VLMPF_abs(pca256_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
    sp32cl256pca0(i, :)=ST_VLMPF_abs(pca0_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);
    
    sp64cl256pca256(i, :)=ST_VLMPF_abs(pca256_desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary);
    sp64cl256pca0(i, :)=ST_VLMPF_abs(pca0_desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary);
    
    
    
    
    v256pca256(i, :)=VLAD_1(pca256_desc, cell_Clusters{1}.vocabulary);
    v256pca0(i, :)=VLAD_1(pca0_desc, cell_Clusters{2}.vocabulary);
    
    fv256pca256(i, :)=mexFisherAssign(pca256_desc', cell_gmmModelName{1})'; 
    fv256pca0(i, :)=mexFisherAssign(pca0_desc', cell_gmmModelName{2})';



         if i == 1
             descParamUsed
         end
         
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

nEncoding=8;
allDist=cell(1, nEncoding);

t_feature=sp32cl256pca256;
t_feature(:, end-(size(cell_spClusters{1}.vocabulary, 1)*size(cell_Clusters{1}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{1}.vocabulary, 1)*size(cell_Clusters{1}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{1}=temp * temp';

t_feature=sp32cl256pca0;
t_feature(:, end-(size(cell_spClusters{1}.vocabulary, 1)*size(cell_Clusters{2}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{1}.vocabulary, 1)*size(cell_Clusters{2}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{2}=temp * temp';

t_feature=sp64cl256pca256;
t_feature(:, end-(size(cell_spClusters{2}.vocabulary, 1)*size(cell_Clusters{1}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{2}.vocabulary, 1)*size(cell_Clusters{1}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{3}=temp * temp';

t_feature=sp64cl256pca0;
t_feature(:, end-(size(cell_spClusters{2}.vocabulary, 1)*size(cell_Clusters{2}.vocabulary, 1)) + 1 :end)= ...
    PowerNormalization(t_feature(:, end-(size(cell_spClusters{2}.vocabulary, 1)*size(cell_Clusters{2}.vocabulary, 1)) +1 :end), 0.5);
temp=NormalizeRowsUnit(t_feature); allDist{4}=temp * temp';


temp=NormalizeRowsUnit(PowerNormalization(v256pca256, alpha)); allDist{5}=temp * temp';
temp=NormalizeRowsUnit(PowerNormalization(v256pca0, alpha)); allDist{6}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(fv256pca256, alpha)); allDist{7}=temp * temp';
temp=NormalizeRowsUnit(PowerNormalization(fv256pca0, alpha)); allDist{8}=temp * temp';

clear temp t_feature

all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


for k=1:nEncoding
k
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

fprintf('Accuracy for encoding %d: %.3f\n',k, mean(mean(cat(2, accuracy{:}))'));

end

delete(gcp('nocreate'))

clear allDist


finalAcc=zeros(1,nEncoding);
for j=1:nEncoding

    finalAcc(j)=mean(mean(cat(2, all_accuracy{j}{:}), 2));
    fprintf('%.3f\n', finalAcc(j));

    
end



 bazeSavePath='/home/ionut/asustor_ionut/Data/results/cvpr2017/ucf50/';
 
 if length(cell_spClusters)==2
     descParam.sp_clDim_check=[size(cell_spClusters{1}.vocabulary, 1) size(cell_spClusters{2}.vocabulary, 1) ]
     descParam.check_Clusters=[size(cell_Clusters{1}.vocabulary, 1) size(cell_Clusters{2}.vocabulary, 1) ]
 end
 
 
 clfsOut_sp32cl256pca0 = all_clfsOut(2);
 acc_sp32cl256pca0 = all_accuracy(2);

saveName = [bazeSavePath 'clfsOut/'  DescParam2Name(descParam) '_PNL2memb__sp32cl256pca0.mat']
save(saveName, '-v7.3', 'descParam', 'clfsOut_sp32cl256pca0', 'acc_sp32cl256pca0');

saveName2 = [bazeSavePath 'videoRep/'  DescParam2Name(descParam) '__sp32cl256pca0.mat']
save(saveName2, '-v7.3', 'descParam', 'sp32cl256pca0');


 clfsOut_sp64cl256pca0 = all_clfsOut(4);
 acc_sp64cl256pca0 = all_accuracy(4);

saveName = [bazeSavePath 'clfsOut/'  DescParam2Name(descParam) '_PNL2memb__sp64cl256pca0.mat']
save(saveName, '-v7.3', 'descParam', 'clfsOut_sp64cl256pca0', 'acc_sp64cl256pca0');

saveName2 = [bazeSavePath 'videoRep/'  DescParam2Name(descParam) '__sp64cl256pca0.mat']
save(saveName2, '-v7.3', 'descParam', 'sp64cl256pca0');



 
saveName = [bazeSavePath 'clfsOut/'  DescParam2Name(descParam) '.mat']
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

saveName2 = [bazeSavePath 'videoRep/'  DescParam2Name(descParam) '.mat']
save(saveName2, '-v7.3', 'descParam', 'sp32cl256pca256', 'sp32cl256pca0', 'sp64cl256pca256', 'sp64cl256pca0', 'v256pca256', 'v256pca0', 'fv256pca256', 'fv256pca0');

 