function [representation]=getRepresentationMultiClusters(desc,bovwCluster, cell_smallCls, func)
representation=[];

%Calculate the similarity using euclidian distance
distance=distmj(desc, bovwCluster.vocabulary);
[~, assign]=min(distance, [], 2);

if strfind(func2str(func), 'stdDiff')>0
    stdDiff=1;
else
    stdDiff=0;
end

for i=1:length(cell_smallCls)
    
    assigned=(assign==i);
    
    if sum(assigned)>0
        if stdDiff==1
            rep=func(desc(assigned, :), cell_smallCls{i}.vocabulary, cell_smallCls{i}.st_d);
        else
            rep=func(desc(assigned, :), cell_smallCls{i}.vocabulary);
        end 
        
    else
        rep=zeros(1, size(cell_smallCls{i}.vocabulary,1)*size(cell_smallCls{i}.vocabulary,2));
    end
    
    
    representation=cat(2, representation, rep);


end