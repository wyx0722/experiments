hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if ~isempty(findstr(hostname, 'cocoa'))
    basePath='/home/ionut/asustor_ionut/'
elseif ~isempty(findstr(hostname, 'Halley'))
      basePath='/home/ionut/asustor_ionut_2/'    
end


typeFeature=cell(1,5);
typeFeature{1}=@FEVidHmgDense
typeFeature{2}=@FEVidHofDense
typeFeature{3}=@FEVidHogDense
typeFeature{4}=@FEVidMBHxDense
typeFeature{5}=@FEVidMBHyDense


fsr=[1 2 3 6];

all_accuracy=cell(length(typeFeature),length(fsr));
all_clfsOut=cell(length(typeFeature),length(fsr));


mType='Vid';

normStrategy='ROOTSIFT';
datasetName='HMDB51';
cl=256;
nPar=5;
alpha=0.5;%!!!!!!!change
d=72;
savePath=[basePath 'Data/results/mtap2017/hmdb51/'];
%fsr=?;





for i=1:length(typeFeature)
    i
    for j=1:length(fsr)
        j
         [all_accuracy{i}{j}, all_clfsOut{i}{j}] = SD_VLADFramework_hmdb51_UCF101(typeFeature{i},mType, normStrategy,datasetName,cl, nPar, alpha, savePath, fsr(j));
    end
end
