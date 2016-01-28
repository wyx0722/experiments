function [] = Test_newNorm_alpha( typeFeature, normStrategy, alpha )

if nargin<3
    alpha=0.1:0.1:0.9
end

for i=1:length(alpha)
    fprintf('Testing for alpha=%.2f ... \n',alpha(i) );
    VLAD256_doClassification_IDT(typeFeature, normStrategy, alpha(i))
end

