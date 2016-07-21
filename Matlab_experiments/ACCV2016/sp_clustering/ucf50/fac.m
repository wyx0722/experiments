function [ rez ] = fac( n )

if n>1
   rez=n*fac(n-1);
else
    rez=1;
end


end

