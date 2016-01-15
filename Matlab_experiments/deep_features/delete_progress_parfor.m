
m=6;



parpool(2);

parfor i=1:m

diff_noPool_conv5_3=diff_noPoolFeatureMaps( '/Users/Ionut/Desktop/work/tempVGG/v_BaseballPitch_g01_c01/conv5_3.txt', 'conv5_3');
fprintf('\b%d\n', i);

end

delete(gcp('nocreate'))