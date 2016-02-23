

videosList = textread('video-list.txt', '%s', 'delimiter', '\n');

for i=1:length(videosList)
    videosList{i}=['/home/ionut/Data/UCF50/Videos/' videosList{i} '.avi'];
end



descParam.BlockSize = [8 8 6];
descParam.NumBlocks = [3 3 2];
descParam.MediaType = 'Vid';
descParam.NumOr = 8;
descParam.FlowMethod = 'Horn-Schunck'; % Horn-Schunk optical opticalFlow

fsr=[1 2 3 6];

nrFrames=zeros(length(fsr),length(videosList));
tElapsed_loadVideo=zeros(length(fsr),length(videosList));
tElapsed_OF=zeros(length(fsr),length(videosList));
tElapsed_hog=zeros(length(fsr),length(videosList));
tElapsed_hsm=zeros(length(fsr),length(videosList));
tElapsed_hof=zeros(length(fsr),length(videosList));
tElapsed_mbhx=zeros(length(fsr),length(videosList));
tElapsed_mbhy=zeros(length(fsr),length(videosList));

tStart_all=tic;


for s=1:length(fsr)
    fprintf('\n frame sampling rate: %d \n', fsr(s));
    for i=1:length(videosList)

        fprintf('%d ', i);

        %%Load video
        tStart_loadVideo=tic;
        video = i_VideoRead(videosList{i});
        video = video(:,:,1:fsr(s):end);
        tElapsed_loadVideo(s,i)=toc(tStart_loadVideo);

        nrFrames(s,i)=size(video, 3);

        %compute optical flow
        tStart_OF=tic;
        opticalFlow = Video2OpticalFlow(video, descParam.FlowMethod);
        tElapsed_OF(s,i)=toc(tStart_OF);

        %extract hog
        tStart_hog=tic;
        [hog_desc, hog_info] = Video2DenseHOGVolumes(video, ...
                                             descParam.BlockSize, ...
                                             descParam.NumBlocks, ...
                                             descParam.NumOr);
         tElapsed_hog(s,i)=toc(tStart_hog);

        %extract hsm
        tStart_hsm=tic;                                
        %simpleMotion=video(:, :, 1:end-1)-video(:, :, 2:end);
        [hsm_desc, hsm_info] = Video2DenseHOGVolumes((video(:, :, 1:end-1)-video(:, :, 2:end)), ...
                                             descParam.BlockSize, ...
                                             descParam.NumBlocks, ...
                                             descParam.NumOr);                                 
        tElapsed_hsm(s,i)=toc(tStart_hsm);                              


        %extract hof
        tStart_hof=tic;
        [hof_desc, hof_info] = Video2DenseHOFVolumes_m(opticalFlow, ...
                                             descParam.BlockSize, ...
                                             descParam.NumBlocks, ...
                                             descParam.NumOr);
         tElapsed_hof(s,i)=toc(tStart_hof);


        %extract mbhx
        tStart_mbhx=tic;
        %xOpticalFlow=real(opticalFlow);
        [mbhx_desc, mbhxinfo] = Video2DenseHOGVolumes(real(opticalFlow), ...
                                             descParam.BlockSize, ...
                                             descParam.NumBlocks, ...
                                             descParam.NumOr); 
         tElapsed_mbhx(s,i)=toc(tStart_mbhx);                               

        %extract mbhy                                 
        tStart_mbhy=tic;                                
        %yOpticalFlow=imag(opticalFlow);
        [mbhy_desc, mbhy_info] = Video2DenseHOGVolumes(imag(opticalFlow), ...
                                             descParam.BlockSize, ...
                                             descParam.NumBlocks, ...
                                             descParam.NumOr); 
        tElapsed_mbhy(s,i)=toc(tStart_mbhy);

    end
end
                                 
tElapsed_all=toc(tStart_all);                                