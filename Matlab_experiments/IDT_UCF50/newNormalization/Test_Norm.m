function [] = Test_Norm(normStrategy)

typeFeature=cell(1, 4);
typeFeature{1}='HOG';
typeFeature{2}='HOF';
typeFeature{3}='MBHx';
typeFeature{4}='MBHy';

for i=1:length(typeFeature)
    fprintf('Testing for normStrategy=%s and typeFeature=%s ... \n',normStrategy, typeFeature{i});
    VLAD256_doClassification_IDT(typeFeature{i}, normStrategy)
end

