function  [all_accuracy, all_clfsOut]  = denseFramework(typeFeature, normStrategy, d, cl_VLAD, cl_FV, fsr, nPar)



global DATAopts;
DATAopts = UCFInit;


addpath('./../');%!!!!!!!

% Parameter settings for descriptor extraction
clear descParam
descParam.Func = typeFeature;
descParam.Normalisation=normStrategy;
descParam.pcaDim = d;
descParam.numClusters_VLAD = cl_VLAD;
descParam.numClusters_FV = cl_FV;

descParam.NumBlocks = [3 3 2];

if nargin>5
descParam.FrameSampleRate = fsr;
descParam.BlockSize = [8 8 6/fsr];
else
    descParam.BlockSize = [8 8 6];
end



descParam.MediaType = 'Vid';
descParam.NumOr = 8;

%descParam.FrameSampleRate = 1;
%descParam.ColourSpace = colourSpace

sRow = [1 3];
sCol = [1 1];





descParam



vocabularyIms = GetVideosPlusLabels('smallEnd');

vocabularyImsPaths=cell(size(vocabularyIms));

for i=1:length(vocabularyImsPaths)
    vocabularyImsPaths{i}=sprintf(DATAopts.videoPath, vocabularyIms{i});
end

%%%%%%%%%%%%%%%%
funcVocab=cell(1,2);
funcVocab{1}=@CreateVocabularyKmeansPca_m;
funcVocab{2}=@CreateVocabularyGMMPca;

allClusters=cell(1,2);
allClusters{1}=descParam.numClusters_VLAD;
allClusters{2}=descParam.numClusters_FV;

vocabs=cell(1,2);
pcaMaps=cell(1,2);

parpool(2);
parfor i=1:length(funcVocab)
    [vocabs{i}, pcaMaps{i}]=funcVocab{i}(vocabularyImsPaths, descParam, allClusters{i}, descParam.pcaDim);
end
delete(gcp('nocreate'))

vocab_vlad256=vocabs{1}{1};
vocab_vlad512=vocabs{1}{2};
vocab_fv=vocabs{2};

pcaMap_vlad=pcaMaps{1};
pcaMap_fv=pcaMaps{2};
%%%%%%%%%%%%%%


                                            
%[vocabulary, pcaMap] = CreateVocabularyKmeansPca(vocabularyImsPaths, descParam, ...
%                                                descParam.numClusters, descParam.pcaDim); 

%make unit for the vocabulary
nV_vocab_vlad256=vocab_vlad256;
nV_vocab_vlad256.vocabulary=NormalizeRowsUnit(nV_vocab_vlad256.vocabulary);
nV_vocab_vlad512=vocab_vlad512;
nV_vocab_vlad512.vocabulary=NormalizeRowsUnit(nV_vocab_vlad512.vocabulary);

% Now create set
[vids, labs, groups] = GetVideosPlusLabels('Full');
fullPathVids=cell(size(vids));

for i=1:length(fullPathVids)
    fullPathVids{i}=sprintf(DATAopts.videoPath, vids{i});
end







[tDesc] = MediaName2Descriptor(fullPathVids{1}, descParam, pcaMap_vlad);

t=VLAD_1_mean(tDesc, vocab_vlad256.vocabulary);
vlad256_1=zeros(length(vids), length(t), 'like', t);
vlad256_2=zeros(length(vids), length(t), 'like', t);
vlad256_3=zeros(length(vids), length(t), 'like', t);
vlad256_4=zeros(length(vids), length(t), 'like', t);

t=VLAD_1_mean_fast(tDesc, nV_vocab_vlad256.vocabulary);
vlad256f_1=zeros(length(vids), length(t), 'like', t);
vlad256f_2=zeros(length(vids), length(t), 'like', t);
vlad256f_3=zeros(length(vids), length(t), 'like', t);
vlad256f_4=zeros(length(vids), length(t), 'like', t);

t=VLAD_1_mean(tDesc, vocab_vlad512.vocabulary);
vlad512_1=zeros(length(vids), length(t), 'like', t);
vlad512_2=zeros(length(vids), length(t), 'like', t);
vlad512_3=zeros(length(vids), length(t), 'like', t);
vlad512_4=zeros(length(vids), length(t), 'like', t);

t=VLAD_1_mean_fast(tDesc, nV_vocab_vlad512.vocabulary);
vlad512f_1=zeros(length(vids), length(t), 'like', t);
vlad512f_2=zeros(length(vids), length(t), 'like', t);
vlad512f_3=zeros(length(vids), length(t), 'like', t);
vlad512f_4=zeros(length(vids), length(t), 'like', t);


t=BoostingVLAD_paper_fast(tDesc, nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
b_f_1=zeros(length(vids), length(t), 'like', t);
b_f_2=zeros(length(vids), length(t), 'like', t);
b_f_3=zeros(length(vids), length(t), 'like', t);
b_f_4=zeros(length(vids), length(t), 'like', t);

t=SD_VLAD(tDesc, vocab_vlad256.vocabulary, vocab_vlad256.st_d);
sd_1=zeros(length(vids), length(t), 'like', t);
sd_2=zeros(length(vids), length(t), 'like', t);
sd_3=zeros(length(vids), length(t), 'like', t);
sd_4=zeros(length(vids), length(t), 'like', t);

t=SD_VLAD_fast(tDesc, nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
sd_f_1=zeros(length(vids), length(t), 'like', t);
sd_f_2=zeros(length(vids), length(t), 'like', t);
sd_f_3=zeros(length(vids), length(t), 'like', t);
sd_f_4=zeros(length(vids), length(t), 'like', t);


t=mexFisherAssign(tDesc', vocab_fv)';
fv_1=zeros(length(vids), length(t), 'like', t);
fv_2=zeros(length(vids), length(t), 'like', t);
fv_3=zeros(length(vids), length(t), 'like', t);
fv_4=zeros(length(vids), length(t), 'like', t);





parpool(nPar);

% Now object visual word frequency histograms
fprintf('Descriptor extraction  for %d vids: ', length(fullPathVids));
parfor i=1:length(fullPathVids)
    fprintf('%d \n', i)
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(fullPathVids{i}, descParam, pcaMap);
    n_desc = NormalizeRowsUnit(desc);
    
        % Feature vector assignment with spatial pyramid
    featSpIdx = SpatialPyramidSeparationIdx(info, sRow, sCol)';
    
    
    vlad256_1(i, :)=VLAD_1_mean(desc(featSpIdx(1,:), :), vocab_vlad256.vocabulary);
    vlad256_2(i, :)=VLAD_1_mean(desc(featSpIdx(2,:), :), vocab_vlad256.vocabulary);
    vlad256_3(i, :)=VLAD_1_mean(desc(featSpIdx(3,:), :), vocab_vlad256.vocabulary);
    vlad256_4(i, :)=VLAD_1_mean(desc(featSpIdx(4,:), :), vocab_vlad256.vocabulary);
    
    vlad256f_1(i, :)=VLAD_1_mean_fast(n_desc(featSpIdx(1,:), :), nV_vocab_vlad256.vocabulary);
    vlad256f_2(i, :)=VLAD_1_mean_fast(n_desc(featSpIdx(2,:), :), nV_vocab_vlad256.vocabulary);
    vlad256f_3(i, :)=VLAD_1_mean_fast(n_desc(featSpIdx(3,:), :), nV_vocab_vlad256.vocabulary);
    vlad256f_4(i, :)=VLAD_1_mean_fast(n_desc(featSpIdx(4,:), :), nV_vocab_vlad256.vocabulary);
    
    vlad512_1(i, :)=VLAD_1_mean(desc(featSpIdx(1,:), :), vocab_vlad512.vocabulary);
    vlad512_2(i, :)=VLAD_1_mean(desc(featSpIdx(2,:), :), vocab_vlad512.vocabulary);
    vlad512_3(i, :)=VLAD_1_mean(desc(featSpIdx(3,:), :), vocab_vlad512.vocabulary);
    vlad512_4(i, :)=VLAD_1_mean(desc(featSpIdx(4,:), :), vocab_vlad512.vocabulary);
    
    vlad512f_1(i, :)=VLAD_1_mean_fast(n_desc(featSpIdx(1,:), :), nV_vocab_vlad512.vocabulary);
    vlad512f_2(i, :)=VLAD_1_mean_fast(n_desc(featSpIdx(2,:), :), nV_vocab_vlad512.vocabulary);
    vlad512f_3(i, :)=VLAD_1_mean_fast(n_desc(featSpIdx(3,:), :), nV_vocab_vlad512.vocabulary);
    vlad512f_4(i, :)=VLAD_1_mean_fast(n_desc(featSpIdx(4,:), :), nV_vocab_vlad512.vocabulary);
    
    b_f_1(i, :) = BoostingVLAD_paper_fast(n_desc(featSpIdx(1,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
    b_f_2(i, :) = BoostingVLAD_paper_fast(n_desc(featSpIdx(2,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
    b_f_3(i, :) = BoostingVLAD_paper_fast(n_desc(featSpIdx(3,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
    b_f_4(i, :) = BoostingVLAD_paper_fast(n_desc(featSpIdx(4,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
    
    sd_1(i, :) = SD_VLAD(desc(featSpIdx(1,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);
    sd_2(i, :) = SD_VLAD(desc(featSpIdx(2,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);
    sd_3(i, :) = SD_VLAD(desc(featSpIdx(3,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);
    sd_4(i, :) = SD_VLAD(desc(featSpIdx(4,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);
    
    sd_f_1(i, :) = SD_VLAD_fast(n_desc(featSpIdx(1,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
    sd_f_2(i, :) = SD_VLAD_fast(n_desc(featSpIdx(2,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
    sd_f_3(i, :) = SD_VLAD_fast(n_desc(featSpIdx(3,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
    sd_f_4(i, :) = SD_VLAD_fast(n_desc(featSpIdx(4,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
    
    
    t_desc=desc';
    fv_1(i, :)=mexFisherAssign(t_desc(:,featSpIdx(1,:)), vocab_fv)';
    fv_2(i, :)=mexFisherAssign(t_desc(:,featSpIdx(2,:)), vocab_fv)';
    fv_3(i, :)=mexFisherAssign(t_desc(:,featSpIdx(3,:)), vocab_fv)';
    fv_4(i, :)=mexFisherAssign(t_desc(:,featSpIdx(4,:)), vocab_fv)';
    
    
        
         if i == 1
             descParamUsed
         end
         
end
fprintf('\nDone!\n');




%% Do classification

nEncoding=9;
allDist=cell(1, nEncoding);

vlad256=cat(2, vlad256_1,vlad256_2, vlad256_3, vlad256_4);
vlad256f= cat(2, vlad256f_1, vlad256f_2, vlad256f_3, vlad256f_4);
vlad512 = cat(2, vlad512_1, vlad512_2, vlad512_3, vlad512_4);
vlad512f = cat(2, vlad512f_1, vlad512f_2, vlad512f_3, vlad512f_4);
b_f = cat(2, b_f_1, b_f_2, b_f_3, b_f_4);
sd = cat(2, sd_1, sd_2, sd_3, sd_4);
sd_f = cat(2, sd_f_1, sd_f_2, sd_f_3, sd_f_4);
fv = cat(2, fv_1, fv_2, fv_3, fv_4);


temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(vlad256, descParam.pcaDim), 0.5));
allDist{1}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(vlad256f, descParam.pcaDim), 0.5));
allDist{2}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(vlad512, descParam.pcaDim), 0.5));
allDist{3}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(vlad512f, descParam.pcaDim), 0.5));
allDist{4}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(b_f, descParam.pcaDim), 0.5));
allDist{5}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(sd, descParam.pcaDim), 0.5));
allDist{6}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(sd_f, descParam.pcaDim), 0.5));
allDist{7}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(fv, 0.5));
allDist{8}=temp * temp';

temp=NormalizeRowsUnit(PowerNormalization(intranormalizationFeatures(fv, descParam.pcaDim), 0.5));
allDist{9}=temp * temp';

clear temp


all_clfsOut=cell(1,nEncoding);
all_accuracy=cell(1,nEncoding);

cRange = 100;
nReps = 1;
nFolds = 3;


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

%descType=func2str(descParam.Func);

hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if ~isempty(findstr(hostname, 'cocoa'))
    rezPath='/home/ionut/asustor_ionut/Data/results/mtap2017/'
else if ~isempty(findstr(hostname, 'Halley'))
      rezPath='/home/ionut/asustor_ionut_2/Data/results/mtap2017/'
    end
end


try    
    
    fileName=[rezPath  'resultsDenseVW.txt'];
    
    fileID=fopen(fileName, 'a');
    
    fprintf(fileID, '%s PNL2 norm before classification (alpha=0.5) \n  vlad256: %.3f   vlad256f: %.3f   vlad512: %.3f   vlad512f: %.3f  \n bost_f: %.3f   SD-VLAD: %.3f   SD-VLAD_d: %.3f   FV: %.3f  FV_intraL2: %.3f', ...
            DescParam2Name(descParam), mean(mean(cat(2, all_accuracy{1}{:}))), mean(mean(cat(2, all_accuracy{2}{:}))), mean(mean(cat(2, all_accuracy{3}{:}))), ...
            mean(mean(cat(2, all_accuracy{4}{:}))), mean(mean(cat(2, all_accuracy{5}{:}))), mean(mean(cat(2, all_accuracy{6}{:}))), ...
            mean(mean(cat(2, all_accuracy{7}{:}))), mean(mean(cat(2, all_accuracy{8}{:}))), mean(mean(cat(2, all_accuracy{9}{:}))) );
    
    fclose(fileID);
    
catch err
    
    fileName='resultsDenseVW.txt';
    fileID=fopen(fileName, 'a');
    
    fprintf(fileID, '%s PNL2 norm before classification (alpha=0.5) \n  vlad256: %.3f   vlad256f: %.3f   vlad512: %.3f   vlad512f: %.3f  \n bost_f: %.3f   SD-VLAD: %.3f   SD-VLAD_d: %.3f   FV: %.3f  FV_intraL2: %.3f', ...
            DescParam2Name(descParam), mean(mean(cat(2, all_accuracy{1}{:}))), mean(mean(cat(2, all_accuracy{2}{:}))), mean(mean(cat(2, all_accuracy{3}{:}))), ...
            mean(mean(cat(2, all_accuracy{4}{:}))), mean(mean(cat(2, all_accuracy{5}{:}))), mean(mean(cat(2, all_accuracy{6}{:}))), ...
            mean(mean(cat(2, all_accuracy{7}{:}))), mean(mean(cat(2, all_accuracy{8}{:}))), mean(mean(cat(2, all_accuracy{9}{:}))) );
    
    fclose(fileID);
    
    warning('error writing %s. Instead the file%s was saved in: ',err, fileName);
        
end

    
saveName = [rezPath 'clfsOut/'  DescParam2Name(descParam) '.mat'];
save(saveName, '-v7.3', 'descParam', 'all_clfsOut', 'all_accuracy');

 saveName2 = [rezPath 'videoRep/' DescParam2Name(descParam) '.mat'];
 save(saveName2, '-v7.3', 'vlad256', 'vlad256f', 'vlad512', 'vlad512f', 'b_f', 'sd', 'sd_f', 'fv');
end
