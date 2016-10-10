
load('/home/ionut/experiments/Matlab_experiments/ACCV2016/sp_clustering/hmdb51/timing_permV.mat');

%nRandomVideos=100;

randVideos=allPathFeatures(permV(1:nRandomVideos));

t_sp32cl64=zeros(1, length(randVideos));
t_sp32cl128=zeros(1, length(randVideos));
t_sp32cl200=zeros(1, length(randVideos));
t_sp32cl256=zeros(1, length(randVideos));
t_sp32cl512=zeros(1, length(randVideos));

t_v64=zeros(1, length(randVideos));
t_v128=zeros(1, length(randVideos));
t_v200=zeros(1, length(randVideos));
t_v256=zeros(1, length(randVideos));
t_v512=zeros(1, length(randVideos));


fprintf('Feature extraction  for %d vids: ', length(randVideos));
%parpool(2);
for i=1:length(randVideos)
    
    if mod(i, 10)==0
        fprintf('%d ', i)%fprintf('%d \n', i)
    end
    % Extract descriptors
    
    [desc, info, descParamUsed] = MediaName2Descriptor(randVideos{i}, descParam, pcaMap);
   
    
    desc=NormalizeRowsUnit(desc);
    info.spInfo=NormalizeRowsUnit(info.spInfo);
    
    tic; ST_VLMPF_abs_inner(desc, cell_Clusters{1}.vocabulary, info.spInfo, cell_spClusters{6}.vocabulary); t_sp32cl64(i)=toc;
    tic; ST_VLMPF_abs_inner(desc, cell_Clusters{2}.vocabulary, info.spInfo, cell_spClusters{6}.vocabulary); t_sp32cl128(i)=toc;
    tic; ST_VLMPF_abs_inner(desc, cell_Clusters{3}.vocabulary, info.spInfo, cell_spClusters{6}.vocabulary); t_sp32cl200(i)=toc;
    tic; ST_VLMPF_abs_inner(desc, cell_Clusters{4}.vocabulary, info.spInfo, cell_spClusters{6}.vocabulary); t_sp32cl256(i)=toc;
    tic; ST_VLMPF_abs_inner(desc, cell_Clusters{5}.vocabulary, info.spInfo, cell_spClusters{6}.vocabulary); t_sp32cl512(i)=toc;

    tic; VLAD_1_fast(desc, cell_Clusters{1}.vocabulary); t_v64(i)=toc;
    tic; VLAD_1_fast(desc, cell_Clusters{2}.vocabulary); t_v128(i)=toc;
    tic; VLAD_1_fast(desc, cell_Clusters{3}.vocabulary); t_v200(i)=toc;
    tic; VLAD_1_fast(desc, cell_Clusters{4}.vocabulary); t_v256(i)=toc;
    tic; VLAD_1_fast(desc, cell_Clusters{5}.vocabulary); t_v512(i)=toc;
    
    
end
%delete(gcp('nocreate'))
fprintf('\nDone!\n');


fprintf('Average time for %d videos t_sp32cl64: %.3f \n', length(randVideos), mean(t_sp32cl64));
fprintf('Average time for %d videos t_sp32cl128: %.3f \n', length(randVideos), mean(t_sp32cl128));
fprintf('Average time for %d videos t_sp32cl200: %.3f \n', length(randVideos), mean(t_sp32cl200));
fprintf('Average time for %d videos t_sp32cl256: %.3f \n', length(randVideos), mean(t_sp32cl256));
fprintf('Average time for %d videos t_sp32cl512: %.3f \n', length(randVideos), mean(t_sp32cl512));

fprintf('Average time for %d videos t_v64: %.3f \n', length(randVideos), mean(t_v64));
fprintf('Average time for %d videos t_v128: %.3f \n', length(randVideos), mean(t_v128));
fprintf('Average time for %d videos t_v200: %.3f \n', length(randVideos), mean(t_v200));
fprintf('Average time for %d videos t_v256: %.3f \n', length(randVideos), mean(t_v256));
fprintf('Average time for %d videos t_v512: %.3f \n', length(randVideos), mean(t_v512));


