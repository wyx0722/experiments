


trial=20;
t_v512=zeros(1, trial);
t_v256=zeros(1, trial);
t_sp32=zeros(1, trial);
t_sp32_fast=zeros(1, trial);
for i=1:trial
    desc=rand(6000,48);
    vocab256=rand(256,48);
    vocab512=rand(512,48);

    spInfo=rand(6000, 3);
    spVocab=rand(32,3);
    
    tic
    v512=VLAD_1_mean(desc, vocab512);
    t_v512(i)=toc;
    
    tic
    v256=VLAD_1_mean(desc, vocab256);
    t_v256(i)=toc;
    
    tic
    sp32=VLAD_1_mean_spClustering_memb(desc, vocab256, spInfo, spVocab);
    t_sp32(i)=toc;
    
    tic
    sp32_fast=fast_VLAD_1_mean_spClustering_memb(desc, vocab256, spInfo, spVocab);
    t_sp32_fast(i)=toc;
    
    
    
end

fprintf('the average time for %d triels for VLAD512: %.3f\n', trial, mean(t_v512));
fprintf('the average time for %d triels for VLAD256: %.3f\n', trial,mean(t_v256));
fprintf('the average time for %d triels for VLAD256+sp32: %.3f\n', trial,mean(t_sp32));
fprintf('the average time for %d triels for *FAST* VLAD256+sp32: %.3f\n', trial,mean(t_sp32_fast));
