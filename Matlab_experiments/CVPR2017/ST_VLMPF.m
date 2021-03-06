function [st_vlmpf_rep]=ST_VLMPF(desc, vocab, spInfo, spVocab)
%VLAD=VLAD(desc, vocab)
%Compute VLAD encoding for a set of descriptors (VLAD_1 is because is assigned only one cluster for each descriptor)

% input:
%    desc: N x D matrix of descriptors (N-number of descriptos, D the dimensionality of descriptors)
%    vocab: K x D visual vocabulary (K - number of visual words (centers), D the dimensionality of each visual word )
%
% output
%    VLAD: 1 x (K * D) VLAD encoding vector.
%
%       Ionut Cosmin Duta - 2015



dimDesc=size(desc, 2); % the dimension of the descriptors
nDesc=size(desc, 1); % number of the descriptors
K=size(vocab, 1);  % the size of the vocabulary



%Calculate the similarity using euclidian distance
distance=distmj(desc, vocab);
[~, assign]=min(distance, [], 2);

%Calculate VLAD for each word of the vocabulary
max_pool=cell(1, K);
memb=zeros(size(desc,1), K);

for i=1:K
    
    assigned=(assign==i); % get the descriptors assigned to the cluster i;
    
    if sum(assigned)>0 % compute VLAD for each visual word (cluster) that has at least one descriptor assigned
        
        %compute the difference between descriptors and visual word (cluster)
        %diff=bsxfun(@minus, desc(assigned, :), vocab(i, :));
        
        
        %Calculate the sum over these differences
        max_pool{i}=max(desc(assigned, :), [], 1);
        memb(assigned, i)=1;
        
        
    else
        % no desccriptor in the cluser then put zeros
        max_pool{i}=zeros(1, dimDesc); 

    end 
        
        
end

%Concatenate all the VLAD vectors for each cluster to crate the final VLAD
%vector
max_pool=cat(2, max_pool{:});
%the assignement for spatial info

spDistance = distmj(spInfo, spVocab);
[~, spAssign] = min(spDistance, [], 2);

spWordMax=cell(1, size(spVocab, 1));
spMemb=cell(1, size(spVocab, 1));
for i=1:size(spVocab, 1)
    spAssigned=(spAssign==i); % get the descriptors assigned to the cluster i;
    nAssigned=sum(spAssigned);
    
    if nAssigned>0
        spWordMax{i}=max(desc(spAssigned, :), [], 1);
        spMemb{i}=sum(memb(spAssigned, :), 1);
    else
        spWordMax{i}=zeros(1, dimDesc);
        spMemb{i}=zeros(1, K);
    end
end

spMax=cat(2, spWordMax{:});
spMemb=cat(2,spMemb{:});

st_vlmpf_rep=cat(2,max_pool, spMax, spMemb);
end