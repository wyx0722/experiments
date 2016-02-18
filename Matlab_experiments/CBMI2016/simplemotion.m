% descParam.nFrames=10;
% 
% 
% simpleMotion_nF = video(:, :, 1:end-descParam.nFrames+1) - video(:, :, 2:end-descParam.nFrames+2);
% for i=2:descParam.nFrames-1
% 
%         simpleMotion_nF = simpleMotion_nF + (video(:, :, 1:end-descParam.nFrames+1) - video(:, :, i+1:end+(i-descParam.nFrames+1)));
% end
% 
% 
% 
% 
% 
% simpleMotion_nF2 = video(:, :, 1:end-descParam.nFrames+1) - video(:, :, 2:end-descParam.nFrames+2);
% for i=2:descParam.nFrames-1
% 
%         simpleMotion_nF2 = simpleMotion_nF2 + (video(:, :, i:end-(descParam.nFrames-i)) - video(:, :, i+1:end-(descParam.nFrames-i-1)));
% end

sm=sm_sfr6;
for i=1:size(sm, 3)
    i
    if i==1
        figure, imshow(sm(:, :, i));
    else
        imshow(sm(:, :, i));
    end
    pause(0.5);
    %pause('on');
    %pause;
end
% 
% 
% 
% for i=1:size(simpleMotion, 3)
%     if i==1
%         figure, imshow(simpleMotion(:, :, i));
%     else
%         imshow(simpleMotion(:, :, i));
%     end
%     pause(0.03);
% end
% 
% 
% 
% 
% for i=1:size(sm_second, 3)
%     if i==1
%         figure, imshow(sm_second(:, :, i));
%     else
%         imshow(sm_second(:, :, i));
%     end
%     pause(0.03);
% end