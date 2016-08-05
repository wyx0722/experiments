function [ enc_vect ] = enc_simVocabs_avg( desc, vocabsClass)

enc_vect=cell(1,length(vocabsClass));

for v=1:length(vocabsClass)
    distance=distmj(desc, vocabsClass{v});
    [distAssign, assign]=min(distance, [], 2);
    
    %check if all the vocabularies are the same size
%     if size(vocabsClass{v}, 1)~=k
%         fprintf('Size vacab%d: %d and size vocab%d: %d \n', 1, k, v,size(vocabsClass{v}, 1));
%         warning('The size of the vocabularies are not equal!!!!!!!!');
%         keyboard
%     end
    
    %get the average distance for each cluster
    k=size(vocabsClass{v}, 1);
    singleVoc=zeros(1, k);
    for i=1:k
        assigned=(assign==i);
        if(sum(assigned)>0)
           singleVoc(i)=mean(distAssign(assigned)); 
        end               
    end
    
    enc_vect{v}=singleVoc;
    
end

enc_vect=cat(2, enc_vect{:});

end


%the old one
% function [ enc_vect ] = enc_simVocabs_avg( desc, vocabsClass)
% 
% k=size(vocabsClass{1}, 1);
% enc_vect=zeros(1,length(vocabsClass)*k);
% 
% for v=1:length(vocabsClass)
%     distance=distmj(desc, vocabsClass{v});
%     [distAssign, assign]=min(distance, [], 2);
%     
%     %check if all the vocabularies are the same size
%     if size(vocabsClass{v}, 1)~=k
%         fprintf('Size vacab%d: %d and size vocab%d: %d \n', 1, k, v,size(vocabsClass{v}, 1));
%         warning('The size of the vocabularies are not equal!!!!!!!!');
%         keyboard
%     end
%     
%     %get the average distance for each cluster
%     for i=1:k
%         assigned=(assign==i);
%         if(sum(assigned)>0)
%             poz=v*k-k+i;
%            enc_vect(poz)=mean(distAssign(assigned)); 
%         end               
%     end
%     
% end
% 
% 
% 
% end
