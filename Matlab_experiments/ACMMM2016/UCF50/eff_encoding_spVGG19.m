global DATAopts;
DATAopts = UCFInit;

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = @FEVid_deepFeatures;
descParam.MediaType = 'DeepF';
descParam.Layer='pool5';
descParam.net='SpVGG19';
descParam.Normalisation='None'; %'ROOTSIFT';

descParam.numClusters = 256;

descParam.pcaDim = 0;

descParam

%%%%%%%%%%
bazePathFeatures='/home/ionut/halley_ionut/Data/VGG_19_v-features_rawFrames_UCF50/Videos/' %change


vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=[bazePathFeatures char(vocabularyIms(i)) '/pool5.txt'];
end


                                           
        [vocabulary, pcaMap] = CreateVocabularyKmeansPca(vocabularyImsPaths, descParam, ...
                                                        descParam.numClusters, descParam.pcaDim);

        [gmmModelName, pcaMap2] = CreateVocabularyGMMPca(vocabularyImsPaths, descParam, ...
                                                        descParam.numClusters, descParam.pcaDim);




%vocabulary = NormalizeRowsUnit(vocabulary); %make unit length

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
pathFeatures=cell(size(vids));

for i=1:length(pathFeatures)
    pathFeatures{i}=[bazePathFeatures char(vids(i)) '/pool5.txt'];
end

%nSample=10;

sample=randperm(length(pathFeatures),nSample );



[tDesc] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
%tDesc=NormalizeRowsUnit(tDesc);
tVLAD=VLAD_1(tDesc, vocabulary);

vladNoMean=zeros(length(sample), length(tVLAD), 'like', tVLAD);    
maxEncode=zeros(length(sample), length(tVLAD), 'like', tVLAD);

[tDesc] = MediaName2Descriptor(pathFeatures{1}, descParam, pcaMap);
FV=mexFisherAssign(tDesc', gmmModelName)';

fisherVectors=zeros(length(sample), length(FV), 'like', FV);

time_vladNoMean= zeros(1, length(sample));  
time_maxEncode= zeros(1, length(sample));
time_fisherVectors= zeros(1, length(sample));

time_DescExtrac=zeros(1, length(sample));
nFrames=zeros(1, length(sample));

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(pathFeatures));
for i=1:length(sample)
    %fprintf('%d \n', i)
    % Extract descriptors
    
    fprintf('%d %s \n', i, pathFeatures{sample(i)});
    
    tStart=tic;
    [desc, info, descParamUsed] = MediaName2Descriptor(pathFeatures{sample(i)}, descParam, pcaMap);
   % desc = NormalizeRowsUnit(desc);
   time_DescExtrac(i)=toc(tStart);
   
   nFrames(i)=size(desc, 1);
   
    tStart=tic;
    vladNoMean(i, :)=VLAD_1(desc, vocabulary);
    time_vladNoMean(i)=toc(tStart);
    
    tStart=tic;
    maxEncode(i, :)=max_pooling(desc, vocabulary);
    time_maxEncode(i)=toc(tStart);
    
    tStart=tic;
    fisherVectors(i,:)=mexFisherAssign(desc', gmmModelName)';
    time_fisherVectors(i)=toc(tStart);
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');


