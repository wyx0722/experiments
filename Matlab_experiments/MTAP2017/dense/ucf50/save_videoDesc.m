

hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if ~isempty(findstr(hostname, 'cocoa'))
    rezPath='/home/ionut/asustor_ionut/Data/results/mtap2017/'
else if ~isempty(findstr(hostname, 'Halley'))
      rezPath='/home/ionut/asustor_ionut_2/Data/results/mtap2017/'
    end
end



descName={'hog', 'hof', 'mbhx', 'mbhy', 'hmg'}
encName={'_sd', '_sd_f', '_vlad256', '_vlad256f', '_vlad512', '_vlad512f', '_b_f'};


nameBase=[rezPath 'videoRep/VLADbased/encVideoReps/']

for dd=1:length(descName)
    for ee=1:length(encName)
        
        if isempty(strfind(encName{ee}, '2'))
            saveName=[nameBase 'FEVid' descName{dd}  'DenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8pcaDim72numClusters_256' encName{ee} '.mat' ]
        else
            saveName=[nameBase 'FEVid' descName{dd}  'DenseBlockSize8_8_6_FrameSampleRate1MediaTypeVidNormalisationROOTSIFTNumBlocks3_3_2_NumOr8pcaDim72numClusters' encName{ee} '.mat']
        end
        
        encVideoRep=eval(sprintf('%s%s', descName{dd},encName{ee}));
        save(saveName, '-v7.3', 'encVideoRep');
        
    end           
end

