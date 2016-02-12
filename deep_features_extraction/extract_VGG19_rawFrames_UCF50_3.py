import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '/home/ionut/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

#check if the model is downloaded
if not os.path.isfile(caffe_root + 'models/VGG_ILSVRC_19/VGG_ILSVRC_19_layers.caffemodel'):
    print("Download the pre-trained CaffeNet model first!!!!!")

#set caffe to GPU mode, specify also the GPU id
caffe.set_device(3)
caffe.set_mode_gpu()

#load the model for the network
net = caffe.Net(caffe_root + 'models/VGG_ILSVRC_19/VGG_ILSVRC_19_layers_deploy.prototxt',
                caffe_root + 'models/VGG_ILSVRC_19/VGG_ILSVRC_19_layers.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

net.blobs['data'].reshape(1,3,224,224)

#create a list with all the vidos from the dataset
with open('/home/ionut/Data/UCF50_tvL1_OpticalFlow/matlabPathsUCF50.txt') as f:
    videoList = f.read().splitlines()
f.close()


rootPathSave='/home/ionut/Data/VGG_19_v-features_rawFrames_UCF50/Videos/' #the root path to save the features
rootRawFrames='/home/ionut/Data/UCF50_tvL1_OpticalFlow/Videos/' #the root path where the raw frames are saved

#extract features for each video from the list
for video in range(3310,4965):
    print 'Features extraction for video (range(3310,4965))', video+1, ': ', videoList[video] 
    
    #for each video save the features layer in a .txt file. Each features layer are saved in a different .txt file, where each line represents the features layer for a frame within the video
    file_fc8=open(rootPathSave + videoList[video] + '/fc8.txt', 'w')
    file_fc7=open(rootPathSave + videoList[video] + '/fc7.txt', 'w')
    file_fc6=open(rootPathSave + videoList[video] + '/fc6.txt', 'w')
    file_conv5_4=open(rootPathSave + videoList[video] + '/conv5_4.txt', 'w')
    file_conv5_1=open(rootPathSave + videoList[video] + '/conv5_1.txt', 'w')
    #file_conv4_3=open(rootPathSave + videoList[video] + '/conv4_3.txt', 'w')
    file_pool5=open(rootPathSave + videoList[video] + '/pool5.txt', 'w')
    file_pool4=open(rootPathSave + videoList[video] + '/pool4.txt', 'w')
    
    #take all the frames for a video
    for file in os.listdir(rootRawFrames + videoList[video] + '/frames/'):
          if file.endswith(".jpg"):
                net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(rootRawFrames + videoList[video] + '/frames/' + file )) # apply the preprocessing operations for each frame
                net.forward() # forward pass for the frame to get the features

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

                #extract conv5_4 features from the network and save them in a .txt file
                feat_conv5_4 = net.blobs['conv5_4'].data[0]
                feat_conv5_4.tofile(file_conv5_4, sep=" ", format="%s")
                file_conv5_4.write('\n')

                #extract conv5_1 features from the network and save them in a .txt file
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
    file_conv5_4.close()
    file_conv5_1.close()
    #file_conv4_3.close()
    file_pool5.close()
    file_pool4.close()