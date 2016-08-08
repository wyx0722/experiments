function [ newRep, distsVocab, predictedClassVocab ] = getPredictedVocab( initRep, vocabs)

elem=size(initRep,1);
startPoz=zeros(1, length(vocabs));

distsVocab=zeros(elem, length(vocabs));
predictedClassVocab=zeros(size(initRep,1),1);

newRep=zeros(size(initRep));

for v=1:elem
    
    poz=1;
    for i=1:length(vocabs)
        startPoz(i)=poz;
        sV=size(vocabs{i},1);
        distsVocab(v,i)=sum(initRep(v,poz:poz+sV-1));
        poz=poz+sV;
    end

    [~, predictedClassVocab(v)]=min(distsVocab(v, :));

    newRep(v, startPoz(predictedClassVocab(v)):startPoz(predictedClassVocab(v))+size(vocabs{predictedClassVocab(v)},1)-1) = ...
    initRep(v, startPoz(predictedClassVocab(v)):startPoz(predictedClassVocab(v))+size(vocabs{predictedClassVocab(v)},1)-1);

end

end

