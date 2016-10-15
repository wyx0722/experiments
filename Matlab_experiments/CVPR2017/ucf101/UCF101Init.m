function DATAopts = UCF101Init
global MYDATADIR

DATAopts.datadir= [MYDATADIR 'UCF-101/'];

DATAopts.featurePath = [DATAopts.datadir 'Features/'];
DATAopts.videoPath = [DATAopts.datadir 'Videos/%s.avi'];
% DATAopts.descriptorPath = [DATAopts.datadir 'UCF50feat/'];
% DATAopts.stipPath = [DATAopts.datadir 'UCF50stip/'];
% DATAopts.stipMatPath = [DATAopts.datadir 'UCF50StipMat/'];
% DATAopts.stipPath = [DATAopts.datadir 'UCF50dense/'];
% DATAopts.stipMatPath = [DATAopts.datadir 'UCF50DenseMat/'];
DATAopts.vocabularyPath = [DATAopts.datadir 'VisualVocabulary/'];
DATAopts.resultsPath = [DATAopts.datadir 'Results/'];

% Get classes
classStruct = dir([DATAopts.datadir 'Videos/']);

for i=3:length(classStruct)
    DATAopts.classes{i-2} = classStruct(i).name;
end

DATAopts.nclasses = length(DATAopts.classes);
