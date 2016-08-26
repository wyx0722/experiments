function xPower=PowerNormalization(x, alpha)

if (nargin<2)
   alpha =0.5;         %alpha=1/7; alpha=0.139;
end

xPower=sign(x).*(abs(x).^alpha);