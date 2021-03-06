function finalVLAD=fast_VLAD_1_mean_spClustering_memb(desc, vocab, spInfo, spVocab)
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
k1=size(vocab, 1);  % the size of the standard VLAD vocabulary
k2=size(spVocab, 1); %the size of ST vocabulary (number of video divisions)


%Calculate the similarity using euclidian distance
distance=distmj(desc, vocab);
[~, assign]=min(distance, [], 2);

finalVLAD=zeros(1, (k1*dimDesc + k2*(dimDesc+k1)));

%the startin pozition of each component (VLAD, residuals, memberships) in
%the final vector
sV=1;
sR=k1*dimDesc+1;
sM=k1*dimDesc + k2*dimDesc +1;

%genareate zeros
zV=zeros(1, dimDesc);
zR=zV;
zM=zeros(1, k1);
%Calculate VLAD for each word of the vocabulary
%wordVLAD=cell(1, k1);



VLADs=zeros(size(desc));
memb=zeros(size(desc,1), 256);

for i=1:k1
    
    assigned=(assign==i); % get the descriptors assigned to the cluster i;
    nAssigned=sum(assigned);
    if nAssigned>0 % compute VLAD for each visual word (cluster) that has at least one descriptor assigned
        
        %compute the difference between descriptors and visual word (cluster)
        diff=bsxfun(@minus, desc(assigned, :), vocab(i, :));
        
        VLADs(assigned, :)=diff;
        memb(assigned, i)=1;
        
        %Calculate the sum over these differences
        %wordVLAD{i}=(1.0/nAssigned)*sum(diff, 1);
        poz=sV+i*dimDesc-dimDesc;
        finalVLAD(poz:poz+dimDesc-1)=(1.0/nAssigned)*sum(diff, 1);
        %wordVLAD{i}=mean(diff, 1);
        
    else
        % no desccriptor in the cluser then put zeros
        poz=sV+i*dimDesc-dimDesc;
        finalVLAD(poz:poz+dimDesc-1)=zV;      
    end 
        
        
end

%Concatenate all the VLAD vectors for each cluster to crate the final VLAD
%vector
%VLAD=cat(2, wordVLAD{:});



%the assignement for spatial info

spDistance = distmj(spInfo, spVocab);
[~, spAssign] = min(spDistance, [], 2);

%spWordVlad=cell(1, size(spVocab, 1));
%spMemb=cell(1, size(spVocab, 1));
for i=1:k2
    spAssigned=(spAssign==i); % get the descriptors assigned to the cluster i;
    nAssigned=sum(spAssigned);
    
    if nAssigned>0
        %spWordVlad{i}=(1.0/nAssigned) * sum(VLADs(spAssigned, :), 1);
        
        poz=sR+i*dimDesc-dimDesc;
        finalVLAD(poz:poz+dimDesc-1)=(1.0/nAssigned) * sum(VLADs(spAssigned, :), 1);
        
        
        %spMemb{i}=sum(memb(spAssigned, :), 1);
        poz=sM+i*k1-k1;
        finalVLAD(poz:poz+k1-1)=sum(memb(spAssigned, :), 1);
    else
        poz=sR+i*dimDesc-dimDesc;
        finalVLAD(poz:poz+dimDesc-1)=zR;
        
        poz=sM+i*k1-k1;
        finalVLAD(poz:poz+k1-1)=zM;
    end
end

% spVLAD=cat(2, spWordVlad{:});
% spMemb=cat(2,spMemb{:});
% finalVLAD=cat(2,VLAD, spVLAD, spMemb);



