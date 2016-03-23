function DATAopts = HMDB51Init
global MYDATADIR

DATAopts.datadir= [MYDATADIR 'HMDB51/'];

DATAopts.featurePath = [DATAopts.datadir 'Features/'];
DATAopts.videoPath = [DATAopts.datadir 'Videos/'];
%DATAopts.mipPath = [DATAopts.datadir 'HMDB51feat/'];
%DATAopts.stipPath = [DATAopts.datadir 'hmdb51_org_stips/'];
%DATAopts.stipMatPath = [DATAopts.datadir 'HMDB51StipMat/'];
DATAopts.vocabularyPath = [DATAopts.datadir 'VisualVocabulary/'];
DATAopts.resultsPath = [DATAopts.datadir 'Results/'];

% Get classes
classStruct = dir([DATAopts.datadir 'Videos/']);

for i=3:length(classStruct)
    DATAopts.classes{i-2} = classStruct(i).name;
end

DATAopts.nclasses = length(DATAopts.classes);
