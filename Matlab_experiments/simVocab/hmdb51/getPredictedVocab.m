function [ newRep, distsVocab, predictedClassVocab ] = getPredictedVocab( initRep, vocabs)

startPoz=zeros(1, length(vocabs));
distsVocab=zeros(1, length(vocabs));
poz=1;
for i=1:length(vocabs)
    startPoz(i)=poz;
    sV=size(vocabs{i},1);
    distsVocab(i)=sum(initRep(poz:poz+sV-1));
    poz=poz+sV;
end

newRep=zeros(size(initRep));
[~, predictedClassVocab]=min(distsVocab);

newRep(startPoz(predictedClassVocab):startPoz(predictedClassVocab)+size(vocabs{predictedClassVocab},1)-1) = ...
initRep(startPoz(predictedClassVocab):startPoz(predictedClassVocab)+size(vocabs{predictedClassVocab},1)-1);

end

