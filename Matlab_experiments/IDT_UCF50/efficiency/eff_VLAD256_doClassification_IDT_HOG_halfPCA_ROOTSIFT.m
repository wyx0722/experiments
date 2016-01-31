% function doClassificationFisher

global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam

%descParam.BlockSize = [8 8 6];
%descParam.NumBlocks = [3 3 2];
%descParam.NumOr = 8;
%descParam.FrameSampleRate = 1;
%descParam.ColourSpace = colourSpace

descParam.Func = @FEVidHOG_IDT;
descParam.IDTfeature='HOG';
descParam.MediaType = 'IDT';
descParam.Normalisation='ROOTSIFT'; % L2 or 'ROOTSIFT'

%sRow = [1 3];
%sCol = [1 1];

if strcmp(descParam.IDTfeature,'HOF')
    sizeDesc=108;
    
elseif  strcmp(descParam.IDTfeature,'HOG') || strcmp(descParam.IDTfeature,'MBHx') || strcmp(descParam.IDTfeature,'MBHy')  
    sizeDesc=96;   
end

% pcaDim & vocabulary size
descParam.pcaDim = sizeDesc/2;
descParam.numClusters = 256;

%%%%%%%%%%
bazePathFeatures='/home/ionut/Features/Features/UCF50/IDT/Videos/'; %change

vocabularyIms = GetVideosPlusLabels('smallEnd');
vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i))];
end
    
%[gmmModelName, pcaMap] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
%                                                numClusters, pcaDim);
[vocabulary, pcaMap, st_d, skew, nElem] = CreateVocabularyKmeansPca_m(vocabularyImsPaths, descParam, ...
                                                descParam.numClusters, descParam.pcaDim);
                                     


% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');

pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i))];
end


useSP=0;  %change to zero if Spatial Pyramid is not used

if (useSP==1)
    nDivImg=sRow(1)*sRow(2)*sCol(1)*sCol(2);
    if nDivImg>1
        nDivImg=nDivImg+1;
    end

    dimVlad = size(vocabulary, 1) * size(vocabulary, 2) * nDivImg;

else
    dimVlad=size(vocabulary, 1) * size(vocabulary, 2);
end


nSample=10;

sample=randperm(length(pathFeatures),nSample );



v1= zeros(length(sample), dimVlad);  
v2= zeros(length(sample), 2*dimVlad);
v3= zeros(length(sample), 3*dimVlad);

v1_L2= zeros(length(sample), dimVlad);  
v2_L2= zeros(length(sample), 2*dimVlad);
v3_L2= zeros(length(sample), 3*dimVlad);

v1_intraL2= zeros(length(sample), dimVlad);  
v2_intraL2= zeros(length(sample), 2*dimVlad);
v3_intraL2= zeros(length(sample), 3*dimVlad);


time_v1= zeros(1, length(sample));  
time_v2= zeros(1, length(sample));
time_v3= zeros(1, length(sample));

time_v1_L2= zeros(1, length(sample));  
time_v2_L2= zeros(1, length(sample));
time_v3_L2= zeros(1, length(sample));

time_v1_intraL2= zeros(1, length(sample));  
time_v2_intraL2= zeros(1, length(sample));
time_v3_intraL2= zeros(1, length(sample));


time_DescExtrac=zeros(1, length(sample));



% Now object visual word frequency histograms
fprintf('IDT VLAD extraction  for %d vids: ', length(sample));


for i=1:length(sample)
    fprintf('%d \n', i)
    % Extract descriptors
    
    tStart=tic;
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{sample(i)}, descParam, pcaMap);
    time_DescExtrac(i)=toc(tStart);
    
    pathFeatures{sample(i)}
    
    
    tStart=tic;
    v1(i, :)=VLAD_1(desc, vocabulary);
    time_v1(i)=toc(tStart);
    
    tStart=tic;
    v2(i, :)=improvedVLAD(desc, vocabulary, st_d, skew, nElem);
    time_v2(i)=toc(tStart);
    
    tStart=tic;
    v3(i, :)=BoostingVLAD_paper(desc, vocabulary, st_d, skew, nElem);
    time_v3(i)=toc(tStart);
    
    
    
    tStart=tic;
    v1_L2(i, :)=VLAD_1_mean_L2(desc, vocabulary);
    time_v1_L2(i)=toc(tStart);
    
    tStart=tic;
    v2_L2(i, :)=improvedVLAD_intraL2(desc, vocabulary, st_d, skew, nElem);
    time_v2_L2(i)=toc(tStart);
    
    tStart=tic;
    v3_L2(i, :)=BoostingVLAD_paper_intraL2(desc, vocabulary, st_d, skew, nElem);
    time_v3_L2(i)=toc(tStart);
    
    
    
    tStart=tic;
    v1_intraL2(i, :)=intranormalizationFeatures(v1, size(vocabulary, 2));
    time_v1_intraL2(i)=toc(tStart);
    
    tStart=tic;
    v2_intraL2(i, :)=intranormalizationFeatures(v2, size(vocabulary, 2));
    time_v2_intraL2(i)=toc(tStart);
    
    tStart=tic;
    v3_intraL2(i, :)=intranormalizationFeatures(v3, size(vocabulary, 2));
    time_v3_intraL2(i)=toc(tStart);

       
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');


