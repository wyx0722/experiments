
global DATAopts;
DATAopts = UCFInit;

load('/home/ionut/Data/UCF50/VisualVocabulary/KmeansHierarchicalFEVidHogDenseBlockSize8_8_6_MediaTypeVidNumBlocks3_3_2_NumOr8Pca72ClustersK4096D1.mat');
kmeansTree_D1=kmeansTree;
minDesc_D1=minDesc;
rangeDesc_D1=rangeDesc;

load('/home/ionut/halley_ionut/Data/UCF50/VisualVocabulary/KmeansFEVidHofDenseBlockSize8_8_6_MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8Pca72Clusters4096.mat');
vocab_optKmeans4096=vocabulary;

load('/home/ionut/halley_ionut/Data/UCF50/VisualVocabulary/KmeansHierarchicalFEVidHSMDenseBlockSize8_8_6_MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8Pca72ClustersK64D2.mat');

load('/home/ionut/halley_ionut/Data/UCF50/VisualVocabulary/FEVidHSMDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8Pca72Trees4Depth10Trials36.mat')
nTrees = length(maps);

gmmModelName='/home/ionut/Data/UCF50/VisualVocabulary/gmmFEVidHSMDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72Pca72Clusters256.gmm';

load('/home/ionut/halley_ionut/Data/UCF50/VisualVocabulary/KmeansFEVidHogDenseBlockSize8_8_6_MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters256pcaDim72Pca72Clusters256.mat');
vocab256=vocabulary;

load('/home/ionut/Data/UCF50/VisualVocabulary/KmeansFEVidHogDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters512pcaDim72Pca72Clusters512.mat');
vocab512=vocabulary;


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
tElapsed_OptKmenas=zeros(1,length(videosList));
tElapsed_hkmenas=zeros(1,length(videosList));
tElapsed_RF=zeros(1,length(videosList));
tElapsed_FV=zeros(1,length(videosList));
tElapsed_VLAD256=zeros(1,length(videosList));
tElapsed_VLAD512_s=zeros(1,length(videosList));
tElapsed_VLAD512_ms=zeros(1,length(videosList));
tElapsed_VLAD512_mf=zeros(1,length(videosList));
tElapsed_VLAD512_dj=zeros(1,length(videosList));
%tElapsed_pca=zeros(1,length(videosList));
tElapsed_fastRF=zeros(1,length(videosList));
tElapsed_hkmenasFlat=zeros(1,length(videosList));


for i=1:length(videosList)

        fprintf('%d ', i);

        %extract hsm
        tStart_hsm=tic;                                
        [hsm_desc, hsm_info, descParamUsed] = MediaName2Descriptor(videosList{i}, descParam, pcaMap);                                      
        tElapsed_hsm(i)=toc(tStart_hsm);
        
        
 
        
%         %apply PCA
%         tStart_pca=tic;
%         hsm_desc=hsm_desc * pcaMap.data.rot;
%         tElapsed_pca(i)=toc(tStart_pca);

        %make unit length
        tStart_unitL=tic;
        unitLength_hsm_desc=NormalizeRowsUnit(hsm_desc);
        tElapsed_unitL(i)=toc(tStart_unitL);
        
        %get the Spatial Pyramid
        featSpIdx = SpatialPyramidSeparationIdx(hsm_info, sRow, sCol);
    
    
        %OptKmeans
        %vSize_k4096 =size(vocab_optKmeans4096, 1);
        
        tStart_optKmeans=tic;
        cbEntry = cell(1,size(featSpIdx,2));
        for spIdx = 1:size(featSpIdx,2)   
            theSim = vocab_optKmeans4096 * unitLength_hsm_desc(featSpIdx(:,spIdx),:)';
            [~, assignment] = max(theSim);
            vSize_k4096 =size(vocab_optKmeans4096, 1);
            cbEntry{spIdx} = mexCountWords(assignment, vSize_k4096);
        end
        tElapsed_OptKmenas(i)=toc(tStart_optKmeans);
        
        %hkmeansFlat
        tStart_hkmeansFlat=tic;
        for spIdx = 1:size(featSpIdx,2)
            rez_hkmeansFlat = KmeansHierarchicalAssignment(hsm_desc(featSpIdx(:,spIdx),:), kmeansTree_D1, minDesc_D1, rangeDesc_D1);
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
