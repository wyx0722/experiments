
function  recursiveMethod(stiva, val, elem)

for i=1:size(val,1)
   if length(stiva)==elem
       if sum(stiva==1)
       fprintf('%.1f ', stiva);
       fprintf('\n');
       %sol=stiva;
       end   
   
      else   
       stiva=cat(2,stiva,val(i));
       if sum(stiva)<=1
          recursiveMethod(stiva, val, elem) 
       end
   end      
end


end
