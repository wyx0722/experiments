function VLAD=BoostingVLAD_paper_intraL2(desc, vocab, st_d, skew, nElem)
%VLAD=VLAD(desc, vocab)

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
wordVLAD1=cell(1, K);
wordVLAD2=cell(1, K);
wordVLAD3=cell(1, K);


for i=1:K
    
    assigned=(assign==i); % get the descriptors assigned to the cluster i;
    nAssigned=sum(assigned);
    if nAssigned>0 % compute VLAD for each visual word (cluster) that has at least one descriptor assigned
        
        %compute the difference between descriptors and visual word (cluster)
        %diff=bsxfun(@minus, desc(assigned, :), vocab(i, :));
        
        
        %Calculate the sum over these differences
        %wordVLAD{i}=sum(diff, 1);
        

          %compute VLAD1 "V"   
          wordVLAD1{i} = nAssigned * (mean(desc(assigned, :), 1) - vocab(i, :));
 
          if nAssigned>1
              
              
              diff=bsxfun(@minus, desc(assigned, :), mean(desc(assigned, :), 1));
              diff2=diff.^2;

              %compute VLAD2 "V_c"
              wordVLAD2{i}=(1./nAssigned)*sum(diff2, 1) - st_d(i, :).^2;

              %compute VLAD3 "V_s"
              wordVLAD3{i}=((1./nAssigned)*sum(diff.^3, 1)) ./ ((1./nAssigned)*sum(diff2, 1)).^(3/2) - skew(i, :);
          
          else 
              %compute VLAD2 "V_c"  
              wordVLAD2{i}=(1./nAssigned)*(desc(assigned, :) - vocab(i, :)) - st_d(i, :).^2;

              %compute VLAD3 "V_s"
              wordVLAD3{i}=((1./nAssigned)*((desc(assigned, :) - vocab(i, :)).^3)) ./ ((1./nAssigned)*((desc(assigned, :) - ((vocab(i, :)))).^2)).^(3.0/2) - skew(i, :);
          end    
     
         %check if is NaN in VLAD3 (possible after the division)
         if sum(isnan(wordVLAD3{i}))>0
            
             wordVLAD3{i}(isnan(wordVLAD3{i}))=0;
         end
 
        
         wordVLAD1{i}=NormalizeRowsUnit(wordVLAD1{i});
         wordVLAD2{i}=NormalizeRowsUnit(wordVLAD2{i});
         wordVLAD3{i}=NormalizeRowsUnit(wordVLAD3{i});
         
         
    else
        % no desccriptor in the cluser then put zeros
        wordVLAD1{i}=zeros(1, dimDesc);
        wordVLAD2{i}=zeros(1, dimDesc);
        wordVLAD3{i}=zeros(1, dimDesc);
    end 
        
        
end

%Concatenate all the VLAD vectors for each cluster to create the final VLAD
%vector
wordVLAD1=cat(2, wordVLAD1{:});
wordVLAD2=cat(2, wordVLAD2{:});
wordVLAD3=cat(2, wordVLAD3{:});

VLAD=cat(2, wordVLAD1, wordVLAD2, wordVLAD3);
