function finalMaxP=abs_maxPooling_memb(desc, vocab, spInfo, spVocab)
%abs_maxPooling_memb=MaxP(desc, vocab, spInfo, spVocab)
%Compute MaxP encoding for a set of descriptors 

% input:
%    desc: N x D matrix of descriptors (N-number of descriptos, D the dimensionality of descriptors)
%    vocab: K x D visual vocabulary (K - number of visual words (centers), D the dimensionality of each visual word )
%

%
%       Ionut Cosmin Duta - 2015



dimDesc=size(desc, 2); % the dimension of the descriptors
nDesc=size(desc, 1); % number of the descriptors
K=size(vocab, 1);  % the size of the vocabulary



%Calculate the similarity using euclidian distance
distance=distmj(desc, vocab);
[~, assign]=min(distance, [], 2);

%Calculate MaxP for each word of the vocabulary
wordMaxP=cell(1, K);

MaxPs=zeros(size(desc));
memb=zeros(size(desc,1), 256);

for i=1:K
    
    assigned=(assign==i); % get the descriptors assigned to the cluster i;
    nAssigned=sum(assigned);
    if nAssigned>0 % compute MaxP for each visual word (cluster) that has at least one descriptor assigned

        memb(assigned, i)=1;
        
        tDesc=desc(assigned, :);
        [~, idx]=max(abs(tDesc), [], 1);
        wordMaxP{i}=tDesc(sub2ind(size(tDesc), idx, 1:size(tDesc,2)));
        
        
    else
        % no desccriptor in the cluser then put zeros
        wordMaxP{i}=zeros(1, dimDesc);      
    end 
        
        
end

%Concatenate all the MaxP vectors for each cluster to crate the final MaxP
%vector
MaxP=cat(2, wordMaxP{:});



%the assignement for spatial info

spDistance = distmj(spInfo, spVocab);
[~, spAssign] = min(spDistance, [], 2);

spWordMaxP=cell(1, size(spVocab, 1));
spMemb=cell(1, size(spVocab, 1));
for i=1:size(spVocab, 1)
    spAssigned=(spAssign==i); % get the descriptors assigned to the cluster i;
    nAssigned=sum(spAssigned);
    
    if nAssigned>0
        tDesc=desc(spAssigned, :);
        [~, idx]=max(abs(tDesc), [], 1);
        spWordMaxP{i}=tDesc(sub2ind(size(tDesc), idx, 1:size(tDesc,2)));
        spMemb{i}=sum(memb(spAssigned, :), 1);
    else
        spWordMaxP{i}=zeros(1, dimDesc);
        spMemb{i}=zeros(1, K);
    end
end

spMaxP=cat(2, spWordMaxP{:});
spMemb=cat(2,spMemb{:});
finalMaxP=cat(2,MaxP, spMaxP, spMemb);



