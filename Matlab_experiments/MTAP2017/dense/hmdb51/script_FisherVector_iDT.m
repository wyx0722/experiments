hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if ~isempty(findstr(hostname, 'cocoa'))
    basePath='/home/ionut/asustor_ionut/'
elseif ~isempty(findstr(hostname, 'Halley'))
      basePath='/home/ionut/asustor_ionut_2/'    
end
addpath('./../');%!!!!!!!


typeFeature=cell(1,4);
typeFeature{1}='HOG'
typeFeature{2}='HOF'
typeFeature{3}='MBHx'
typeFeature{4}='MBHy'



all_accuracy=cell(length(typeFeature),1);
all_clfsOut=cell(length(typeFeature),1);


mType='IDT';

normStrategy='ROOTSIFT';
datasetName='HMDB51';
cl=256;
nPar=5;
alpha=0.1%!!!!!!!change
fu= @FEVid_IDT;




%!!!!!!!change both!!!!!!!!!!!!!!!!!!!!!
savePath=[basePath 'Data/results/mtap2017/hmdb51/%s/iDT/'];
bazePathFeatures=[basePath 'Data/iDT_Features_HMDB51/Videos/']







for i=1:length(typeFeature)
    i
    
    if ~isempty(strfind(typeFeature{i},'HOF'))
            sizeDesc=108;   
    else%if  strfind(descParam.IDTfeature,'HOG')>0 || strfind(descParam.IDTfeature,'MBHx')>0 || strfind(descParam.IDTfeature,'MBHy')>0  
            sizeDesc=96;   
    end
    
    d=sizeDesc/2;
    
   [all_accuracy{i}, all_clfsOut{i}] = FisherVectorFramework_hmdb51_UCF101__noSpatialPyramid(fu,mType, normStrategy,datasetName,cl, nPar, alpha, savePath, '',bazePathFeatures, d, typeFeature{i});
 
end
