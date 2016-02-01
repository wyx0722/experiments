
poz=randperm(size(hogDesc, 1), 256);

dim=48;

vocab=hogDesc(poz, 1:dim);
desc=hogDesc(:,1:dim);

t_start=tic;
distance1=distmj(desc, vocab);
[~, assign1]=min(distance1, [], 2);
t_stop1=toc(t_start)


n_vocab=NormalizeRowsUnit(vocab);

t_start=tic;
n_desc=NormalizeRowsUnit(desc);
theSim=n_desc * n_vocab';
[~, assign2] = max(theSim, [], 2);
t_stop2=toc(t_start)


% 
% t_stop1 =
% 
%     0.5967
% 
% 
% t_stop2 =
% 
%     0.2081
%     
%     but assign1 and assign2 are not equal, it is necessarily to test