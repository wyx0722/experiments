function [ representation ] = VMGF( features )

dimFeatures=size(features, 2);
[~, assign]=max(features, [], 2);

rep=cell(1,dimFeatures);
ze=zeros(1,dimFeatures);

for i=1:dimFeatures
    assigned=(assign==i);
    
    if sum(assigned)>0
        rep{i}=max(features(assigned, :), [], 1);   
    else
        rep{i}=ze;
    end
    
end

representation=cat(2, rep{:});


end

