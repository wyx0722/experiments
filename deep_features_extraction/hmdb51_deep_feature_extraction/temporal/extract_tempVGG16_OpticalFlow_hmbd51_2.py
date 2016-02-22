
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np


from scipy import misc



# Make sure that caffe is on the python path:
caffe_root = '/home/ionut/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

import os
if not os.path.isfile(caffe_root + 'models/action_recognition/cuhk_action_temporal_vgg_16_split1.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")

#caffe.set_mode_cpu()

caffe.set_device(2)
caffe.set_mode_gpu()

net = caffe.Net(caffe_root + 'models/action_recognition/cuhk_action_temporal_vgg_16_flow_deploy.prototxt',
                caffe_root + 'models/action_recognition/cuhk_action_temporal_vgg_16_split1.caffemodel',
                caffe.TEST)

net.blobs['data'].reshape(1,20,224,224)



#create a list with all the vidos from the dataset
with open('/home/ionut/Data/HMDB51/listVideos.txt') as f:
    videoList = f.read().splitlines()
f.close()



rootPathSave='/home/ionut/Data/hmdb51_action_temporal_vgg_16_split1_features_opticalFlow_tvL1/Videos/' #the root path to save the features
rootRawFrames='/home/ionut/Data/HMDB51_tvL1_OpticalFlow/Videos/' #the root path where the raw frames are saved

stackFrames=10;
sizeImputNet=224;
mean_flow=128

arrayStackFrames=np.zeros((2*stackFrames, sizeImputNet, sizeImputNet))
#extract features for each video from the list
for video in range(1700,3400):
	print 'Features extraction for video (range(1700,3400))', video+1,':  ', videoList[video]

	#for each video save the features layer in a .txt file. Each features layer are saved in a different .txt file, where each line represents the features layer for a frame within the video
	file_fc8=open(rootPathSave + videoList[video] + '/fc8.txt', 'w')
	file_fc7=open(rootPathSave + videoList[video] + '/fc7.txt', 'w')
	file_fc6=open(rootPathSave + videoList[video] + '/fc6.txt', 'w')
	file_conv5_3=open(rootPathSave + videoList[video] + '/conv5_3.txt', 'w')
	file_conv5_1=open(rootPathSave + videoList[video] + '/conv5_1.txt', 'w')
	#file_conv4_3=open(rootPathSave + videoList[video] + '/conv4_3.txt', 'w')
	file_pool5=open(rootPathSave + videoList[video] + '/pool5.txt', 'w')
	file_pool4=open(rootPathSave + videoList[video] + '/pool4.txt', 'w')
	
	nFrames=0;

	for file in os.listdir(rootRawFrames + videoList[video] + '/x_flow/'):
    		if file.endswith(".jpg"):
          		if nFrames<stackFrames:
          			x_temp=misc.imread(rootRawFrames + videoList[video] + '/x_flow/' + file)
          			y_temp=misc.imread(rootRawFrames + videoList[video] + '/y_flow/' + file)
				
          			x_temp=misc.imresize(x_temp, [sizeImputNet, sizeImputNet])
          			y_temp=misc.imresize(y_temp, [sizeImputNet, sizeImputNet])

          			arrayStackFrames[2*nFrames]=x_temp
          			arrayStackFrames[2*nFrames+1]=y_temp
          			nFrames=nFrames+1

          		

          			if nFrames==stackFrames:
          				net.blobs['data'].data[0]=arrayStackFrames - mean_flow
          				net.forward()

          				#extract fc8 features from the network and save them in a .txt file
		              		feat_fc8 = net.blobs['fc8'].data[0]
		              		np.savetxt(file_fc8, feat_fc8, fmt='%s', newline=' ', delimiter=',')
		              		file_fc8.write('\n')

		                  	#extract fc7 features from the network and save them in a .txt file
		              		feat_fc7 = net.blobs['fc7'].data[0]
		              		np.savetxt(file_fc7, feat_fc7, fmt='%s', newline=' ', delimiter=',')
		              		file_fc7.write('\n')

		                  	#extract fc6 features from the network and save them in a .txt file
		              		feat_fc6 = net.blobs['fc6'].data[0]
		              		np.savetxt(file_fc6, feat_fc6, fmt='%s', newline=' ', delimiter=',')
		              		file_fc6.write('\n')

		                  	#extract conv5_3 features from the network and save them in a .txt file
		              		feat_conv5_3 = net.blobs['conv5_3'].data[0]
		              		feat_conv5_3.tofile(file_conv5_3, sep=" ", format="%s")
		              		file_conv5_3.write('\n')
					
		                  	#extract conv4_3 features from the network and save them in a .txt file
		              		feat_conv5_1 = net.blobs['conv5_1'].data[0]
		              		feat_conv5_1.tofile(file_conv5_1, sep=" ", format="%s")
		              		file_conv5_1.write('\n')

					#extract pool5 features from the network and save them in a .txt file
					feat_pool5 = net.blobs['pool5'].data[0]
					feat_pool5.tofile(file_pool5, sep=" ", format="%s")
					file_pool5.write('\n')

					#extract pool4 features from the network and save them in a .txt file
					feat_pool4 = net.blobs['pool4'].data[0]
					feat_pool4.tofile(file_pool4, sep=" ", format="%s")
					file_pool4.write('\n')			

          		else:
          			x_temp=misc.imread(rootRawFrames + videoList[video] + '/x_flow/' + file)
          			y_temp=misc.imread(rootRawFrames + videoList[video] + '/y_flow/' + file)

          			x_temp=misc.imresize(x_temp, [sizeImputNet, sizeImputNet])
          			y_temp=misc.imresize(y_temp, [sizeImputNet, sizeImputNet])

          			arrayStackFrames[0:18]=arrayStackFrames[2:20]
          			arrayStackFrames[18]=x_temp
          			arrayStackFrames[19]=y_temp

          			net.blobs['data'].data[0]=arrayStackFrames - mean_flow
          			net.forward()

				#extract fc8 features from the network and save them in a .txt file
		                feat_fc8 = net.blobs['fc8'].data[0]
		                np.savetxt(file_fc8, feat_fc8, fmt='%s', newline=' ', delimiter=',')
		                file_fc8.write('\n')
		
		                #extract fc7 features from the network and save them in a .txt file
		                feat_fc7 = net.blobs['fc7'].data[0]
		                np.savetxt(file_fc7, feat_fc7, fmt='%s', newline=' ', delimiter=',')
		                file_fc7.write('\n')
		
		                #extract fc6 features from the network and save them in a .txt file
		                feat_fc6 = net.blobs['fc6'].data[0]
		                np.savetxt(file_fc6, feat_fc6, fmt='%s', newline=' ', delimiter=',')
		                file_fc6.write('\n')
		
		                #extract conv5_3 features from the network and save them in a .txt file
		                feat_conv5_3 = net.blobs['conv5_3'].data[0]
		                feat_conv5_3.tofile(file_conv5_3, sep=" ", format="%s")
		                file_conv5_3.write('\n')
		        
		                #extract conv4_3 features from the network and save them in a .txt file
		                feat_conv5_1 = net.blobs['conv5_1'].data[0]
		                feat_conv5_1.tofile(file_conv5_1, sep=" ", format="%s")
		                file_conv5_1.write('\n')
		
		                #extract pool5 features from the network and save them in a .txt file
		                feat_pool5 = net.blobs['pool5'].data[0]
		                feat_pool5.tofile(file_pool5, sep=" ", format="%s")
		                file_pool5.write('\n')
		
		                #extract pool4 features from the network and save them in a .txt file
		                feat_pool4 = net.blobs['pool4'].data[0]
		                feat_pool4.tofile(file_pool4, sep=" ", format="%s")
		                file_pool4.write('\n')

	#close all the opened files
	file_fc8.close()
	file_fc7.close()
	file_fc6.close()
	file_conv5_3.close()
	file_conv5_1.close()
	#file_conv4_3.close()
	file_pool5.close()
  	file_pool4.close()









          		





