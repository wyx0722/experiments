function [idx] = SpatialPyramid_Rows(infoTraj, nRows)


 idx=false(size(infoTraj, 1), nRows+1);
% 
 idx(:, 1)=1;
 
 
 for i=1:nRows
     if i==1
     idx(:, i+1)=infoTraj(:, 3)<=240/nRows;
         
     else
         idx(:, i+1)=infoTraj(:, 3)>(i-1)*(240/nRows) & infoTraj(:, 3)<=i*(240/nRows);       
     end
      
 end
 

