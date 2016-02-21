
tStart_all=tic;

descParam.BlockSize = [8 8 6];
descParam.NumBlocks = [3 3 2];
descParam.MediaType = 'Vid';
descParam.NumOr = 8;
descParam.FlowMethod = 'Horn-Schunck'; % Horn-Schunk optical opticalFlow


%%Load video
tStart_loadVideo=tic;
video = i_VideoRead(videoName);
tElapsed_loadVideo=toc(tStart_loadVideo)


%compute optical flow
tStart_OF=tic;
opticalFlow = Video2OpticalFlow(video, descParam.FlowMethod);
tElapsed_OF=toc(tStart_OF)

%extract hog
tStart_hog=tic;
[hog_desc, hog_info] = Video2DenseHOGVolumes(video, ...
                                     descParam.BlockSize, ...
                                     descParam.NumBlocks, ...
                                     descParam.NumOr);
 tElapsed_hog=toc(tStart_hog)
 
%extract hsm
tStart_hsm=tic;                                
simpleMotion=video(:, :, 1:end-1)-video(:, :, 2:end);
[hsm_desc, hsm_info] = Video2DenseHOGVolumes(simpleMotion, ...
                                     descParam.BlockSize, ...
                                     descParam.NumBlocks, ...
                                     descParam.NumOr);                                 
tElapsed_hsm=toc(tStart_hsm)                              


%extract hof
tStart_hof=tic;
[hof_desc, hof_info] = Video2DenseHOFVolumes_m(opticalFlow, ...
                                     descParam.BlockSize, ...
                                     descParam.NumBlocks, ...
                                     descParam.NumOr);
 tElapsed_hof=toc(tStart_hof)
 
 
%extract mbhx
tStart_mbhx=tic;
xOpticalFlow=real(opticalFlow);
[mbhx_desc, mbhxinfo] = Video2DenseHOGVolumes(xOpticalFlow, ...
                                     descParam.BlockSize, ...
                                     descParam.NumBlocks, ...
                                     descParam.NumOr); 
 tElapsed_mbhx=toc(tStart_mbhx)                                
                                 
%extract mbhy                                 
tStart_mbhy=tic;                                
yOpticalFlow=imag(opticalFlow);
[mbhy_desc, mbhy_info] = Video2DenseHOGVolumes(yOpticalFlow, ...
                                     descParam.BlockSize, ...
                                     descParam.NumBlocks, ...
                                     descParam.NumOr); 
tElapsed_mbhy=toc(tStart_mbhy)
                                 
tElapsed_all=toc(tStart_all)                                