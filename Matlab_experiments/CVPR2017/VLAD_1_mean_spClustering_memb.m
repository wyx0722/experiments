function finalVLAD=VLAD_1_mean_spClustering_memb(desc, vocab, spInfo, spVocab)
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
wordVLAD=cell(1, K);

VLADs=zeros(size(desc));
memb=zeros(size(desc,1), K);

for i=1:K
    
    assigned=(assign==i); % get the descriptors assigned to the cluster i;
    nAssigned=sum(assigned);
    if nAssigned>0 % compute VLAD for each visual word (cluster) that has at least one descriptor assigned
        
        %compute the difference between descriptors and visual word (cluster)
        diff=bsxfun(@minus, desc(assigned, :), vocab(i, :));
        
        VLADs(assigned, :)=diff;
        memb(assigned, i)=1;
        
        %Calculate the sum over these differences
        wordVLAD{i}=(1.0/nAssigned)*sum(diff, 1);
        %wordVLAD{i}=mean(diff, 1);
        
    else
        % no desccriptor in the cluser then put zeros
        wordVLAD{i}=zeros(1, dimDesc);      
    end 
        
        
end

%Concatenate all the VLAD vectors for each cluster to crate the final VLAD
%vector
VLAD=cat(2, wordVLAD{:});



%the assignement for spatial info

spDistance = distmj(spInfo, spVocab);
[~, spAssign] = min(spDistance, [], 2);

spWordVlad=cell(1, size(spVocab, 1));
spMemb=cell(1, size(spVocab, 1));
for i=1:size(spVocab, 1)
    spAssigned=(spAssign==i); % get the descriptors assigned to the cluster i;
    nAssigned=sum(spAssigned);
    
    if nAssigned>0
        spWordVlad{i}=(1.0/nAssigned) * sum(VLADs(spAssigned, :), 1);
        spMemb{i}=sum(memb(spAssigned, :), 1);
    else
        spWordVlad{i}=zeros(1, dimDesc);
        spMemb{i}=zeros(1, K);
    end
end

spVLAD=cat(2, spWordVlad{:});
spMemb=cat(2,spMemb{:});
finalVLAD=cat(2,VLAD, spVLAD, spMemb);



