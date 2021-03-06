addpath('./../');%!!!!!!!
addpath('./../../');%!!!!!!!


global DATAopts;

datasetName='HMDB51';
mType='DeepF';
typeFeature=@FEVid_deepFeatures;
normStrategy='None';
layer='pool5'
Net='SpVGG19';

d=0;

alpha=0.5;
nPar=5;
pathFeatures='/home/ionut/asustor_ionut/Data/hmdb51_VGG_19_features_rawFrames/Videos/'%%%%%channge~~~~~~~~~~~~


descParam.Dataset=datasetName;
descParam.MediaType=mType;
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;

descParam.Clusters=[64 128 200 256 512];
descParam.spClusters=[2 4 8 16 20 32 64 128 256];


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
        bazePathFeatures=pathFeatures
        
    case 'DeepF'
        descParam.Layer=layer;
        descParam.net=Net;
 
        sizeDesc=512;
        if exist('d', 'var')
            descParam.pcaDim=d;
        else
            descParam.pcaDim=sizeDesc/2;
        end
        bazePathFeatures=pathFeatures
end


descParam



if ~isempty(strfind(descParam.Dataset, 'HMDB51'))
    DATAopts = HMDB51Init;
    [allVids, labs, splits] = GetVideosPlusLabels();
elseif ~isempty(strfind(descParam.Dataset, 'UCF101'))
    DATAopts = UCF101Init;
    [allVids, labs, splits] = GetVideosPlusLabels('Challenge');
end

%create the full path of the fetures for each video
allPathFeatures=cell(size(allVids));
for i=1:size(allVids, 1)
    
    if ~isempty(strfind(descParam.MediaType, 'DeepF'))
        
        if ~isempty(strfind(descParam.net, 'C3D'))
            file_extension='.mat';
        else
            file_extension='.txt';
        end
            
        allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4) '/' descParam.Layer file_extension];
        
    elseif ~isempty(strfind(descParam.MediaType, 'IDT'))
        allPathFeatures{i}=[bazePathFeatures allVids{i}(1:end-4)];
    elseif ~isempty(strfind(descParam.MediaType, 'Vid'))
        
        if ~isempty(strfind(descParam.Dataset, 'HMDB51'))
            allPathFeatures{i}=[DATAopts.videoPath, allVids{i}];
        else
            allPathFeatures{i}=sprintf(DATAopts.videoPath, allVids{i});
        end
    end
end

vocabularyPathFeatures=allPathFeatures(1:4:end);%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[cell_Clusters, cell_spClusters, pcaMap] = CreateVocabularyKmeansPca_sptCl(vocabularyPathFeatures, descParam);
 
[tDesc, info] = MediaName2Descriptor(allPathFeatures{1}, descParam, pcaMap);                                           


%all_rep=cell(1, 16);%!!!!!!!!


t=ST_VLMPF_abs(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); sp2cl200=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); sp4cl200=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); sp8cl200=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{4}.vocabulary); sp16cl200=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary); sp20cl200=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{7}.vocabulary); sp64cl200=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{8}.vocabulary); sp128cl200=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(tDesc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{9}.vocabulary); sp256cl200=zeros(length(allPathFeatures), length(t), 'like', t);

t=ST_VLMPF_abs(tDesc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); sp2cl256=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); sp4cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(tDesc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); sp8cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(tDesc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{4}.vocabulary); sp16cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(tDesc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary); sp20cl256=zeros(length(allPathFeatures), length(t), 'like', t); 
t=ST_VLMPF_abs(tDesc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{7}.vocabulary); sp64cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(tDesc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{8}.vocabulary); sp128cl256=zeros(length(allPathFeatures), length(t), 'like', t);
t=ST_VLMPF_abs(tDesc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{9}.vocabulary); sp256cl256=zeros(length(allPathFeatures), length(t), 'like', t);


nDesc=zeros(1, length(allPathFeatures));

%parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(allPathFeatures));
for i=1:length(allPathFeatures)%parfor i=1:length(allPathFeatures)
    if mod(i, 100)==0
        fprintf('%d ', i)%fprintf('%d \n', i)
    end
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(allPathFeatures{i}, descParam, pcaMap);
    nDesc(i)=size(desc,1);
    
    
    
    sp2cl200(i, :)=ST_VLMPF_abs(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary); 
    sp4cl200(i, :)=ST_VLMPF_abs(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); 
    sp8cl200(i, :)=ST_VLMPF_abs(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); 
    sp16cl200(i, :)=ST_VLMPF_abs(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{4}.vocabulary); 
    sp20cl200(i, :)=ST_VLMPF_abs(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary); 
    sp64cl200(i, :)=ST_VLMPF_abs(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{7}.vocabulary); 
    sp128cl200(i, :)=ST_VLMPF_abs(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{8}.vocabulary); 
    sp256cl200(i, :)=ST_VLMPF_abs(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{9}.vocabulary); 

    sp2cl256(i, :)=ST_VLMPF_abs(desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{1}.vocabulary);  
    sp4cl256(i, :)=ST_VLMPF_abs(desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); 
    sp8cl256(i, :)=ST_VLMPF_abs(desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{2}.vocabulary); 
    sp16cl256(i, :)=ST_VLMPF_abs(desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{4}.vocabulary); 
    sp20cl256(i, :)=ST_VLMPF_abs(desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{5}.vocabulary);  
    sp64cl256(i, :)=ST_VLMPF_abs(desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{7}.vocabulary); 
    sp128cl256(i, :)=ST_VLMPF_abs(desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{8}.vocabulary); 
    sp256cl256(i, :)=ST_VLMPF_abs(desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{9}.vocabulary);

         if i == 1
             descParamUsed
         end
         
end
delete(gcp('nocreate'))
fprintf('\nDone!\n');

nEncoding=16;
allDist=cell(1, nEncoding);

temp=NormalizeRowsUnit(sp2cl200); allDist{1}=temp * temp';
temp=NormalizeRowsUnit(sp4cl200); allDist{2}=temp * temp';
temp=NormalizeRowsUnit(sp8cl200); allDist{3}=temp * temp';
temp=NormalizeRowsUnit(sp16cl200); allDist{4}=temp * temp';
temp=NormalizeRowsUnit(sp20cl200); allDist{5}=temp * temp';
temp=NormalizeRowsUnit(sp64cl200); allDist{6}=temp * temp';
temp=NormalizeRowsUnit(sp128cl200); allDist{7}=temp * temp';
temp=NormalizeRowsUnit(sp256cl200); allDist{8}=temp * temp';


temp=NormalizeRowsUnit(sp2cl256); allDist{9}=temp * temp';
temp=NormalizeRowsUnit(sp4cl256); allDist{10}=temp * temp';
temp=NormalizeRowsUnit(sp8cl256); allDist{11}=temp * temp';
temp=NormalizeRowsUnit(sp16cl256); allDist{12}=temp * temp';
temp=NormalizeRowsUnit(sp20cl256); allDist{13}=temp * temp';
temp=NormalizeRowsUnit(sp64cl256); allDist{14}=temp * temp';
temp=NormalizeRowsUnit(sp128cl256); allDist{15}=temp * temp';
temp=NormalizeRowsUnit(sp256cl256); allDist{16}=temp * temp';


%each row for the cell represents the results for all 3 splits
all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);
clfsOut=cell(1,nEncoding);
accuracy=cell(1,nEncoding);
%mean_all_clfsOut=cell(nEncoding,1);
mean_all_accuracy=cell(nEncoding,1);

cRange = 100;
nReps = 1;
nFolds = 3;



parpool(3);
%%
for k=1:nEncoding
    k
    parfor i=1:3
        
        trainI = splits(:,i) == 1;
        
       if ~isempty(strfind(datasetName, 'HMDB51'))
            testI  = splits(:,i) == 2;
       elseif ~isempty(strfind(datasetName, 'UCF101'))
            testI=~trainI;
       end
       
        trainLabs = labs(trainI,:);
        testLabs = labs(testI,:);
        
        trainDist = allDist{k}(trainI, trainI);
        testDist = allDist{k}(testI, trainI);
        

        [~, clfsOut{i}] = SvmPKOpt(trainDist, testDist, trainLabs, testLabs, cRange, nReps, nFolds);
        accuracy{i} = ClassificationAccuracy(clfsOut{i}, testLabs);
        %fprintf('accuracy: %.3f\n', accuracy);
    end
     all_clfsOut{k}=clfsOut;
     all_accuracy{k}=accuracy;
end

delete(gcp('nocreate')) %///
%%%%

finalAcc=zeros(1,nEncoding);
for j=1:nEncoding
    %mean_all_clfsOut{j}=(all_clfsOut{j}{1} + all_clfsOut{j}{2} + all_clfsOut{j}{3})./3;
    mean_all_accuracy{j}=(all_accuracy{j}{1} + all_accuracy{j}{2} + all_accuracy{j}{3})./3;
    
    finalAcc(j)=mean(mean_all_accuracy{j});
    fprintf('%.3f\n', finalAcc(j));
end