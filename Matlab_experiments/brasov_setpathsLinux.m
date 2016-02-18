basedir = '/data/ionut/Matlab/';% % CHANGE!

addpath([basedir 'BagOfWords']);
addpath([basedir 'Misc']);
addpath([basedir 'Datasets/UCF50']);
addpath([basedir 'Datasets']);
addpath([basedir 'Blobs']);
addpath([basedir '.hg']);
addpath([basedir 'Mex/anigaussm']);
addpath([basedir 'Mex']);
addpath([basedir '/Mex/Deprecated']);
addpath([basedir 'Mex/FelzenSegment']);
addpath([basedir 'Mex/MatrixMultiplication']);
addpath([basedir 'SelectiveSearch']);
addpath([basedir 'SimilarityMeasures']);
addpath([basedir 'Statistics']);
addpath([basedir 'FeatureExtraction']);
addpath([basedir 'Toolboxes']);
addpath([basedir 'Toolboxes/prtools4.1.3']);
addpath([basedir 'FeatureExtraction/SourceRealtimeSiftSurfRelease']);
addpath([basedir 'Datasets/Voc']);
addpath([basedir 'Toolboxes/libsvm-mat-2.89-3']);
addpath([basedir 'Toolboxes/dd_tools']);
addpath([basedir 'Toolboxes/gmm_fisher_jorge']);


global MYDATADIR
MYDATADIR = '/data/ionut/Data/'
global DATAopts;
DATAopts = UCFInit

addpath([basedir 'Toolboxes/vlfeat-0.9.14/toolbox/kmeans']);
addpath([basedir 'Toolboxes/vlfeat-0.9.14/toolbox/imop']);
addpath([basedir '/Toolboxes/vlfeat-0.9.14/toolbox/sift']);
addpath([basedir '/Toolboxes/vlfeat-0.9.14/toolbox/misc']);
addpath([basedir '/Toolboxes/vlfeat-0.9.14/toolbox/mex/mexa64']);

addpath('/data/ionut/Read_Video');
addpath('/data/ionut/Read_Video/mmread/');

addpath('/data/ionut/experiments/Matlab_experiments/versions_VLAD/')


