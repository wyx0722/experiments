function [ solutions ] = getSolutions( val, elem)

sizeInterval=length(val);
stiva{1}=1;
i=1;
poz=1;
while ~isempty(stiva)
    
    if size(stiva,2)==elem
        s=sumStiva(stiva, val);
        if s==1
          solutions{i}=cell2mat(stiva);
          stiva=stiva(1:end-1);
          
          if stiva{end}<sizeInterval
            stiva(end)=stiva{end}+1;
          else stiva=stiva(1:end-1);
          end
          
        elseif s>1
           stiva=stiva(1:end-1);
           
           if stiva{end}<sizeInterval
                stiva(end)=stiva{end}+1;
           else stiva=stiva(1:end-1);
           end
          
        else
            stiva(end)=stiva{end}+1;
        end
    elseif 
        
    end
        
    
    
end





% propose=nchoosek(interval, elem)
% 
% right=propose(sum(propose,2)==1, :)
% 
% solutions=cell(size(right,1), elem);
% 
% for i=1:size(right,1)
%     solutions{i}=perms(right(i, :));
% end
% solutions=cell2mat(solutions);

end

function s=sumStiva(stiva, val)
s=0;
for i=1:size(stiva,2)
    s=s+val(stiva{i});
end
end