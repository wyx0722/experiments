function [idx] = SpatialPyramid_RowsMaps(spInfo, nRows)


 idx=false(size(spInfo, 1), nRows+1);
% 
 idx(:, 1)=1;
 
 maxRows=max(spInfo(:,1));
 div=floor(maxRows/3);
 
idx(:, 2)=spInfo(:,1)<=div;
idx(:, 3)=spInfo(:,1)>div & spInfo(:,1)<= maxRows-div;
idx(:, 4)=spInfo(:,1)>maxRows-div;

 
 
end
