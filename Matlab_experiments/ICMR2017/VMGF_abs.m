function [ representation ] = VMGF_abs( desc )

dimFeatures=size(desc, 2);
 
[~, assign]=max(abs(desc), [], 2);

rep=cell(1,dimFeatures);
ze=zeros(1,dimFeatures);

for i=1:dimFeatures
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

