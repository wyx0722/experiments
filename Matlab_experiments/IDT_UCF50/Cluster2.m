function [idx, emptyCluster] = Cluster2(infoTraj, nrCl)


% idx=zeros(size(infoTraj, 1), nrCl+1);
% 
% idx(:, 1)=1;

emptyCluster=0;

try
    idx=kmeans(infoTraj(:, 2:3), nrCl);
    
catch
    emptyCluster=1;
    
    idx=randi([1 nrCl], size(infoTraj, 1));
end

