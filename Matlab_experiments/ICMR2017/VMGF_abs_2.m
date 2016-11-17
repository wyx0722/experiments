function [ representation ] = VMGF_abs_2( desc, initialDesc )


 
[~, assign]=max(abs(initialDesc), [], 2);


rep=cell(1,size(initialDesc, 2));
ze=zeros(1,size(desc, 2));

for i=1:size(initialDesc, 2)
    assigned=(assign==i);
    
    if sum(assigned)>0
        
         tDesc=desc(assigned, :);
        [~, idx]=max(abs(tDesc), [], 1);
        rep{i}=tDesc(sub2ind(size(tDesc), idx, 1:size(tDesc,2)));     
        %rep{i}=max(desc(assigned, :), [], 1);   
    else
        rep{i}=ze;
    end
    
end

representation=cat(2, rep{:});


end

