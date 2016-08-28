

hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if ~isempty(findstr(hostname, 'cocoa'))
    rezPath='/home/ionut/asustor_ionut/Data/results/mtap2017/'
else if ~isempty(findstr(hostname, 'Halley'))
      rezPath='/home/ionut/asustor_ionut_2/Data/results/mtap2017/'
    end
end



load([rezPath 'videoRep/FEVidHmgDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters_FV256numClusters_VLAD256_512_pcaDim72.mat']);
hmg_vlad256=vlad256;
hmg_vlad256f=vlad256f;
hmg_vlad512=vlad512;
hmg_vlad512f=vlad512f;
hmg_b_f=b_f;
hmg_sd=sd;
hmg_sd_f=sd_f;



load([rezPath 'FEVidHofDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters_FV256numClusters_VLAD256_512_pcaDim72.mat']);
hof_vlad256=vlad256;
hof_vlad256f=vlad256f;
hof_vlad512=vlad512;
hof_vlad512f=vlad512f;
hof_b_f=b_f;
hof_sd=sd;
hof_sd_f=sd_f;


load([rezPath 'FEVidHogDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters_FV256numClusters_VLAD256_512_pcaDim72.mat']);
hog_vlad256=vlad256;
hog_vlad256f=vlad256f;
hog_vlad512=vlad512;
hog_vlad512f=vlad512f;
hog_b_f=b_f;
hog_sd=sd;
hog_sd_f=sd_f;

load([rezPath 'FEVidMBHxDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters_FV256numClusters_VLAD256_512_pcaDim72.mat']);
mbhx_vlad256=vlad256;
mbhx_vlad256f=vlad256f;
mbhx_vlad512=vlad512;
mbhx_vlad512f=vlad512f;
mbhx_b_f=b_f;
mbhx_sd=sd;
mbhx_sd_f=sd_f;


load([rezPath 'FEVidMBHyDenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8numClusters_FV256numClusters_VLAD256_512_pcaDim72.mat']);
mbhy_vlad256=vlad256;
mbhy_vlad256f=vlad256f;
mbhy_vlad512=vlad512;
mbhy_vlad512f=vlad512f;
mbhy_b_f=b_f;
mbhy_sd=sd;
mbhy_sd_f=sd_f;

clear vlad256 vlad256f vlad512 vlad512f b_f sd sd_f fv 

