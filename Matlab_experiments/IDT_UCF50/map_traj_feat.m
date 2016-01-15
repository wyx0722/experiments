
addpath(genpath('/data/MM22/lanzhzh/code/toolbox/vlfeat-0.9.18//'));

path = '/data/MM3/lanzhzh/MIFS/data/hmdb51/results/';

while(1)
list = dir(path);

for i = 3:length(list);
	folder = list(i).name;
	imdb_name = [path,'/',folder,'/imdb.mat'];
	encoder_name = [path,'/',folder,'/encoder.mat'];
	cacheDir = [path,'/',folder,'/cache/'];
	if(~exist(imdb_name)|| ~exist(encoder_name));
		continue;
	else
		load(imdb_name);
		encoder = load(encoder_name);
		disp(encoder);
		disp(isempty(encoder));
		if(~isfield(encoder,'type'))
			continue;
		end
		encodeVideo(encoder, images.name, ...
		  'cacheDir', cacheDir) ;
	end
end
end
