
global DATAopts;
DATAopts = UCFInit;



hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if ~isempty(findstr(hostname, 'cocoa'))
    rezPath='/home/ionut/asustor_ionut/'
else if ~isempty(findstr(hostname, 'Halley'))
      rezPath='/home/ionut/asustor_ionut_2/'
    end
end


load([rezPath 'Data/UCF50/VisualVocabulary/KmeansFEVidHmgDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters_FV256numClusters_VLAD256_512_pcaDim72Pca72Clusters256_512.mat']);

gmmModelName='/home/ionut/Data/UCF50/VisualVocabulary/gmmFEVidHmgDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72Pca72Clusters256.gmm';

vocab_vlad256=vocabs{1};
vocab_vlad512=vocabs{2};

nV_vocab_vlad256=vocab_vlad256;
nV_vocab_vlad256.vocabulary=NormalizeRowsUnit(nV_vocab_vlad256.vocabulary);
nV_vocab_vlad512=vocab_vlad512;
nV_vocab_vlad512.vocabulary=NormalizeRowsUnit(nV_vocab_vlad512.vocabulary);



videosList = textread('video-list.txt', '%s', 'delimiter', '\n');

for i=1:length(videosList)
    videosList{i}=['/home/ionut/Data/UCF50/Videos/' videosList{i} '.avi'];
end


descParam.Func = @FEVidHmgDense;
descParam.BlockSize = [8 8 6];
descParam.NumBlocks = [3 3 2];
descParam.MediaType = 'Vid';
descParam.NumOr = 8;
descParam.FlowMethod = 'Horn-Schunck'; % Horn-Schunk optical opticalFlow
sRow = [1 3];
sCol = [1 1];












tElapsed_hsm=zeros(1,length(videosList));
tElapsed_unitL=zeros(1,length(videosList));


tElapsed_VLAD256=zeros(1,length(videosList));
tElapsed_VLAD256_f=zeros(1,length(videosList));
tElapsed_VLAD512=zeros(1,length(videosList));
tElapsed_VLAD512_f=zeros(1,length(videosList));
tElapsed_sd_vlad=zeros(1,length(videosList));
tElapsed_sd_vlad_f=zeros(1,length(videosList));
tElapsed_b=zeros(1,length(videosList));

tElapsed_fv=zeros(1,length(videosList));


for i=1:length(videosList)

        fprintf('%d ', i);

        %extract hsm
        tStart_hsm=tic;                                
        [desc, hsm_info, descParamUsed] = MediaName2Descriptor(videosList{i}, descParam, pcaMap);                                      
        tElapsed_hsm(i)=toc(tStart_hsm);
        
        
 
        
%         %apply PCA
%         tStart_pca=tic;
%         hsm_desc=hsm_desc * pcaMap.data.rot;
%         tElapsed_pca(i)=toc(tStart_pca);

        %make unit length
        tStart_unitL=tic;
        n_desc=NormalizeRowsUnit(desc);
        tElapsed_unitL(i)=toc(tStart_unitL);
        
        %get the Spatial Pyramid
        featSpIdx = SpatialPyramidSeparationIdx(hsm_info, sRow, sCol);
       
            
        tStart_VLAD256=tic;
        vlad256_1 =VLAD_1_mean(desc(featSpIdx(1,:), :), vocab_vlad256.vocabulary);
        vlad256_2 =VLAD_1_mean(desc(featSpIdx(2,:), :), vocab_vlad256.vocabulary);
        vlad256_3 =VLAD_1_mean(desc(featSpIdx(3,:), :), vocab_vlad256.vocabulary);
        vlad256_4 =VLAD_1_mean(desc(featSpIdx(4,:), :), vocab_vlad256.vocabulary);
        tElapsed_VLAD256(i)=toc(tStart_VLAD256);
        
        tStart_VLAD256_f=tic;
        vlad256f_1 =VLAD_1_mean_fast(n_desc(featSpIdx(1,:), :), nV_vocab_vlad256.vocabulary);
        vlad256f_2 =VLAD_1_mean_fast(n_desc(featSpIdx(2,:), :), nV_vocab_vlad256.vocabulary);
        vlad256f_3 =VLAD_1_mean_fast(n_desc(featSpIdx(3,:), :), nV_vocab_vlad256.vocabulary);
        vlad256f_4 =VLAD_1_mean_fast(n_desc(featSpIdx(4,:), :), nV_vocab_vlad256.vocabulary);
        tElapsed_VLAD256_f(i)=toc(tStart_VLAD256_f);
        
        tStart_VLAD512=tic;
        vlad512_1 =VLAD_1_mean(desc(featSpIdx(1,:), :), vocab_vlad512.vocabulary);
        vlad512_2 =VLAD_1_mean(desc(featSpIdx(2,:), :), vocab_vlad512.vocabulary);
        vlad512_3 =VLAD_1_mean(desc(featSpIdx(3,:), :), vocab_vlad512.vocabulary);
        vlad512_4 =VLAD_1_mean(desc(featSpIdx(4,:), :), vocab_vlad512.vocabulary);
        tElapsed_VLAD512(i)=toc(tStart_VLAD512);
        
        tStart_VLAD512_f=tic;
        vlad512f_1 =VLAD_1_mean_fast(n_desc(featSpIdx(1,:), :), nV_vocab_vlad512.vocabulary);
        vlad512f_2 =VLAD_1_mean_fast(n_desc(featSpIdx(2,:), :), nV_vocab_vlad512.vocabulary);
        vlad512f_3 =VLAD_1_mean_fast(n_desc(featSpIdx(3,:), :), nV_vocab_vlad512.vocabulary);
        vlad512f_4 =VLAD_1_mean_fast(n_desc(featSpIdx(4,:), :), nV_vocab_vlad512.vocabulary);
        tElapsed_VLAD512_f(i)=toc(tStart_VLAD512_f);
        
        tStart_b=tic;
        b_f_1  = BoostingVLAD_paper_fast(n_desc(featSpIdx(1,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
        b_f_2  = BoostingVLAD_paper_fast(n_desc(featSpIdx(2,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
        b_f_3  = BoostingVLAD_paper_fast(n_desc(featSpIdx(3,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
        b_f_4  = BoostingVLAD_paper_fast(n_desc(featSpIdx(4,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
        tElapsed_b(i)=toc(tStart_b);
        
        tStart_sd_vlad=tic;
        sd_1  = SD_VLAD(desc(featSpIdx(1,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);
        sd_2  = SD_VLAD(desc(featSpIdx(2,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);
        sd_3  = SD_VLAD(desc(featSpIdx(3,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);
        sd_4  = SD_VLAD(desc(featSpIdx(4,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);
        tElapsed_sd_vlad(i)=toc(tStart_sd_vlad);
        
        tStart_sd_vlad_f=tic;
        sd_f_1  = SD_VLAD_fast(n_desc(featSpIdx(1,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
        sd_f_2  = SD_VLAD_fast(n_desc(featSpIdx(2,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
        sd_f_3  = SD_VLAD_fast(n_desc(featSpIdx(3,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
        sd_f_4  = SD_VLAD_fast(n_desc(featSpIdx(4,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
        tElapsed_sd_vlad_f(i)=toc(tStart_sd_vlad_f);
        
        
        %Fisher Vector
       t_featSpIdx=featSpIdx';
       t_desc=desc';
       
       tStart_fv=tic;
       fisher1=mexFisherAssign(t_desc(:,t_featSpIdx(1,:)), gmmModelName)';
       fisher2=mexFisherAssign(t_desc(:,t_featSpIdx(2,:)), gmmModelName)';
       fisher3=mexFisherAssign(t_desc(:,t_featSpIdx(3,:)), gmmModelName)';
       fisher4=mexFisherAssign(t_desc(:,t_featSpIdx(4,:)), gmmModelName)';
       tElapsed_fv(i)=toc(tStart_fv);
         
        
end
