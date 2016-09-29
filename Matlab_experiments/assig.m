

desc=rand(5000, 72);
vocab=rand(256, 72);

e_dist=distmj(NormalizeRowsUnit(desc), NormalizeRowsUnit(vocab));
[~, assign_e_dist]=min(e_dist, [], 2);

inner_p=NormalizeRowsUnit(desc) * NormalizeRowsUnit(vocab)';
[~, assign_inner_p]=max(inner_p, [], 2);

isequal(assign_e_dist,assign_inner_p)