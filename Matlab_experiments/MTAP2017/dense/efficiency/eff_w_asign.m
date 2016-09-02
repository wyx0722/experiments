
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

vocab_vlad256=vocabs{1};
vocab_vlad512=vocabs{2};

nV_vocab_vlad256=vocab_vlad256;
nV_vocab_vlad256.vocabulary=NormalizeRowsUnit(nV_vocab_vlad256.vocabulary);
nV_vocab_vlad512=vocab_vlad512;
nV_vocab_vlad512.vocabulary=NormalizeRowsUnit(nV_vocab_vlad512.vocabulary);


pcaMap



load('/home/ionut/Data/UCF50/VisualVocabulary/KmeansHierarchicalFEVidHogDenseBlockSize8_8_6_MediaTypeVidNumBlocks3_3_2_NumOr8Pca72ClustersK4096D1.mat');
kmeansTree_D1=kmeansTree;
minDesc_D1=minDesc;
rangeDesc_D1=rangeDesc;



videosList = textread('video-list.txt', '%s', 'delimiter', '\n');

for i=1:length(videosList)
    videosList{i}=['/home/ionut/Data/UCF50/Videos/' videosList{i} '.avi'];
end


descParam.Func = @FEVidHSMDense;
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

        vlad512_1 =VLAD_1_mean(desc(featSpIdx(1,:), :), vocab_vlad512.vocabulary);
        vlad512_2 =VLAD_1_mean(desc(featSpIdx(2,:), :), vocab_vlad512.vocabulary);
        vlad512_3 =VLAD_1_mean(desc(featSpIdx(3,:), :), vocab_vlad512.vocabulary);
        vlad512_4 =VLAD_1_mean(desc(featSpIdx(4,:), :), vocab_vlad512.vocabulary);

        vlad512f_1 =VLAD_1_mean_fast(n_desc(featSpIdx(1,:), :), nV_vocab_vlad512.vocabulary);
        vlad512f_2 =VLAD_1_mean_fast(n_desc(featSpIdx(2,:), :), nV_vocab_vlad512.vocabulary);
        vlad512f_3 =VLAD_1_mean_fast(n_desc(featSpIdx(3,:), :), nV_vocab_vlad512.vocabulary);
        vlad512f_4 =VLAD_1_mean_fast(n_desc(featSpIdx(4,:), :), nV_vocab_vlad512.vocabulary);

        b_f_1  = BoostingVLAD_paper_fast(n_desc(featSpIdx(1,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
        b_f_2  = BoostingVLAD_paper_fast(n_desc(featSpIdx(2,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
        b_f_3  = BoostingVLAD_paper_fast(n_desc(featSpIdx(3,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);
        b_f_4  = BoostingVLAD_paper_fast(n_desc(featSpIdx(4,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d, nV_vocab_vlad256.skew);

        sd_1  = SD_VLAD(desc(featSpIdx(1,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);
        sd_2  = SD_VLAD(desc(featSpIdx(2,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);
        sd_3  = SD_VLAD(desc(featSpIdx(3,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);
        sd_4  = SD_VLAD(desc(featSpIdx(4,:), :), vocab_vlad256.vocabulary, vocab_vlad256.st_d);

        sd_f_1  = SD_VLAD_fast(n_desc(featSpIdx(1,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
        sd_f_2  = SD_VLAD_fast(n_desc(featSpIdx(2,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
        sd_f_3  = SD_VLAD_fast(n_desc(featSpIdx(3,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);
        sd_f_4  = SD_VLAD_fast(n_desc(featSpIdx(4,:), :), nV_vocab_vlad256.vocabulary, nV_vocab_vlad256.st_d);

        
        
        
        
        
        
        
        
        
        
    
        %OptKmeans
        %vSize_k4096 =size(vocab_optKmeans4096, 1);
        
        tStart_optKmeans=tic;
        cbEntry = cell(1,size(featSpIdx,2));
        for spIdx = 1:size(featSpIdx,2)   
            theSim = vocab_optKmeans4096 * n_desc(featSpIdx(:,spIdx),:)';
            [~, assignment] = max(theSim);
            vSize_k4096 =size(vocab_optKmeans4096, 1);
            cbEntry{spIdx} = mexCountWords(assignment, vSize_k4096);
        end
        tElapsed_OptKmenas(i)=toc(tStart_optKmeans);
        
        %hkmeansFlat
        tStart_hkmeansFlat=tic;
        for spIdx = 1:size(featSpIdx,2)
            rez_hkmeansFlat = KmeansHierarchicalAssignment(desc(featSpIdx(:,spIdx),:), kmeansTree_D1, minDesc_D1, rangeDesc_D1);
        end
        tElapsed_hkmenasFlat(i)=toc(tStart_hkmeansFlat);

        
        
%         %hkmeans
%         tStart_hkmeans=tic;
%         for spIdx = 1:size(featSpIdx,2)
%             rez_hkmeans = KmeansHierarchicalAssignment(hsm_desc(featSpIdx(:,spIdx),:), kmeansTree, minDesc, rangeDesc);
%         end
%         tElapsed_hkmenas(i)=toc(tStart_hkmeans);
        
%         
%         %Random Forest
%         t_hsm_desc = hsm_desc'; % We need transposed descriptors
%         tStart_RF=tic;
%         for spI=1:size(featSpIdx,2)
%             for j=1:nTrees
%                 rez_RF = mexTreeAssign(t_hsm_desc(:,featSpIdx(:,spI)), maps{j}, boundaries{j});
%             end
%         end
%         tElapsed_RF(i)=toc(tStart_RF);
        
        
        t_featSpIdx=featSpIdx';
        t_hsm_desc = desc'; % We need transposed descriptors
        %fast Random Forest
        tStart_fastRF=tic;
        for j=1:nTrees
		rez_fastRF = mexTreeAssignSp(t_hsm_desc, t_featSpIdx, maps{j}, boundaries{j});
        end
        tElapsed_fastRF(i)=toc(tStart_fastRF);
        
        
%        %Fisher Vector
%        t_featSpIdx=featSpIdx';
%        
%        tStart_FV=tic;
%        fisher1=mexFisherAssign(t_hsm_desc(:,t_featSpIdx(1,:)), gmmModelName)';
%        fisher2=mexFisherAssign(t_hsm_desc(:,t_featSpIdx(2,:)), gmmModelName)';
%        fisher3=mexFisherAssign(t_hsm_desc(:,t_featSpIdx(3,:)), gmmModelName)';
%        fisher4=mexFisherAssign(t_hsm_desc(:,t_featSpIdx(4,:)), gmmModelName)';
%        tElapsed_FV(i)=toc(tStart_FV);
%        
%        
%        %VLAD256 slow
%        tStart_VLAD256=tic;
%         vlad256_1=VLAD_1_slow(hsm_desc(t_featSpIdx(1,:), :), vocab256);
%         vlad256_2=VLAD_1_slow(hsm_desc(t_featSpIdx(2,:), :), vocab256);
%         vlad256_3=VLAD_1_slow(hsm_desc(t_featSpIdx(3,:), :), vocab256);
%         vlad256_4=VLAD_1_slow(hsm_desc(t_featSpIdx(4,:), :), vocab256);
%         tElapsed_VLAD256(i)=toc(tStart_VLAD256);
%         
%         
%         %VLAD512 slow
%        tStart_VLAD512_s=tic;
%         vlad512_s_1=VLAD_1_slow(hsm_desc(t_featSpIdx(1,:), :), vocab512);
%         vlad512_s_2=VLAD_1_slow(hsm_desc(t_featSpIdx(2,:), :), vocab512);
%         vlad512_s_3=VLAD_1_slow(hsm_desc(t_featSpIdx(3,:), :), vocab512);
%         vlad512_s_4=VLAD_1_slow(hsm_desc(t_featSpIdx(4,:), :), vocab512);
%         tElapsed_VLAD512_s(i)=toc(tStart_VLAD512_s);
%         
%         
%         %VLAD512 mean slow
%          tStart_VLAD512_ms=tic;
%         vlad512_ms_1=VLAD_1_mean_slow(hsm_desc(t_featSpIdx(1,:), :), vocab512);
%         vlad512_ms_2=VLAD_1_mean_slow(hsm_desc(t_featSpIdx(2,:), :), vocab512);
%         vlad512_ms_3=VLAD_1_mean_slow(hsm_desc(t_featSpIdx(3,:), :), vocab512);
%         vlad512_ms_4=VLAD_1_mean_slow(hsm_desc(t_featSpIdx(4,:), :), vocab512);
%         tElapsed_VLAD512_ms(i)=toc(tStart_VLAD512_ms);
%         
%         
%         %VLAD512 mean fast
%         vocab512_ul = NormalizeRowsUnit(vocab512);
%          tStart_VLAD512_mf=tic;
%         vlad512_mf_1=VLAD_1_mean_fast(unitLength_hsm_desc(t_featSpIdx(1,:), :), vocab512_ul);
%         vlad512_mf_2=VLAD_1_mean_fast(unitLength_hsm_desc(t_featSpIdx(2,:), :), vocab512_ul);
%         vlad512_mf_3=VLAD_1_mean_fast(unitLength_hsm_desc(t_featSpIdx(3,:), :), vocab512_ul);
%         vlad512_mf_4=VLAD_1_mean_fast(unitLength_hsm_desc(t_featSpIdx(4,:), :), vocab512_ul);
%         tElapsed_VLAD512_mf(i)=toc(tStart_VLAD512_mf);
%         
%         
%         %VLAD512 using distmj
%         tStart_VLAD512_dj=tic;
%         vlad512_dj_1=VLAD_1_mean(hsm_desc(t_featSpIdx(1,:), :), vocab512);
%         vlad512_dj_2=VLAD_1_mean(hsm_desc(t_featSpIdx(2,:), :), vocab512);
%         vlad512_dj_3=VLAD_1_mean(hsm_desc(t_featSpIdx(3,:), :), vocab512);
%         vlad512_dj_4=VLAD_1_mean(hsm_desc(t_featSpIdx(4,:), :), vocab512);
%         tElapsed_VLAD512_dj(i)=toc(tStart_VLAD512_dj);
%         
        
        
end
