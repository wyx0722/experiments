#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/gpu/gpu.hpp"
//#include "opencv2/core/cuda.hpp"
//#include "opencv2/cudalegacy.hpp"

//#include "opencv2/cudaoptflow.hpp"
//#include "opencv2/cudaarithm.hpp"

#include <ctime>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
using namespace cv;
//using namespace cv::cuda;
using namespace std;

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
       double lowerBound, double higherBound) {
	#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flow_x.rows; ++i) {
		for (int j = 0; j < flow_y.cols; ++j) {
			float x = flow_x.at<float>(i,j);
			float y = flow_y.at<float>(i,j);
			img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
			img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
		}
	}
	#undef CAST
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color){
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

int main(int argc, char** argv){
	// IO operation
	const String keys =
			"{ f vidFile  | ex2.avi | filename of video }"
			"{ x xFlowFile  | tmp/flow_x/ | filename of flow x component }"
			"{ y yFlowfile  | tmp/flow_y/ | filename of flow x component }"
			"{ i imgFile  | tmp/flow_i/ | filename of flow image}"
			"{ b bound  | 15 | specify the maximum of optical flow}"
			"{ t type  | 0 | specify the optical flow algorithm }"
			"{ d device_id  | 0  | set gpu id}"
			"{ s step  | 1 | specify the step for frame sampling}";

	CommandLineParser cmd(argc, argv, keys);
	String vidFile = cmd.get<String>("f");
	String xFlowFile = cmd.get<String>("x");
	String yFlowFile = cmd.get<String>("y");
	String imgFile = cmd.get<String>("i");
	int bound = cmd.get<int>("b");
    int type  = cmd.get<int>("t");
    int device_id = cmd.get<int>("d");
    int step = cmd.get<int>("s");
	time_t tstart, tend;
	
//    cout << vidFile << endl;
//	        cout<<endl<<"vidFile: "<<vidFile<<endl;
        cout<<endl<<"xFlowFile: "<<xFlowFile<<endl;
        cout<<endl<<"yFlowFile: "<<yFlowFile<<endl;
        cout<<endl<<"imgFile: "<<imgFile<<endl;
        cout<<endl<<"bound: "<<bound<<endl;
        cout<<endl<<"type: "<<type<<endl;
        cout<<endl<<"device_id: "<<device_id<<endl;
        cout<<endl<<"step: "<<step<<endl;


	//begin create the list of videos for extractiong optical flow
	
	//the path for the list with videos
	ifstream relativePathVideos ("/home/ionut/Data/UCF50_tvL1_OpticalFlow/matlabPathsUCF50.txt");	
	
	//the path for the dataset of videos
	string pathRootDataset ="/home/ionut/Data/UCF50/Videos/";
	
	//the root path for saving the frames and optical flow
	string pathRootSave="/home/ionut/Data/UCF50_tvL1_OpticalFlow/Videos/";
	
	//the full path for a video from the dataset
	string fullPathVideo;
	
	//the full path for saving the frames and optical flow
	string fullPathSave;
	
	//the list with the path for all the videos for dataset
	vector<string> listVideos;
	
	//the list root for saving the frames and optical flow
	vector<string> listRootSave;
	
	//a line from the file
	string line;
	
	//the folders for saving the frames and optical flow
	string saveFrame="frames/";
	string saveFlow_x="x_flow/";
	string saveFlow_y="y_flow/";

	//the full path for frames and optical flow 
	string pathSaveFrame, pathSaveFlow_x, pathSaveFlow_y;
	
	//auto  dir;
	//const char* c;

	if (relativePathVideos.is_open())
	{
		while(getline(relativePathVideos, line)){
			fullPathVideo = pathRootDataset + line + ".avi";
			listVideos.push_back(fullPathVideo); 
			
			fullPathSave = pathRootSave + line + "/";
			listRootSave.push_back(fullPathSave);

		}
		relativePathVideos.close();
	}
	else cout<<endl<<"Unable to open the file \n";
/*
	for (int i=0; i<listVideos.size(); i++){
		cout<<endl<<listVideos.at(i)<<endl<<listRootSave.at(i)<<endl<<i<<endl;
	}
*/

	//end create the list of videos for extractiong optical flow


	int frame_num = 0;
	
	//Mat image, prev_image, prev_grey, grey, frame, flow_x, flow_y;
	Mat image, prev_image, prev_grey, grey, frame;
/*	
	//GpuMat frame_0, frame_1, flow_u, flow_v;
	GpuMat frame_0, frame_1;
	GpuMat d_flow;

	setDevice(device_id);
	//cuda::FarnebackOpticalFlow alg_farn;
	Ptr<cuda::FarnebackOpticalFlow> alg_farn = cuda::FarnebackOpticalFlow::create();

	//cuda::OpticalFlowDual_TVL1_GPU alg_tvl1;
	 Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();
	//cuda::BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);
	Ptr<cuda::BroxOpticalFlow> alg_brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
//        cout <<endl<< "It took for CUDA init: "<< difftime(tend, tstart) <<" second(s)."<< endl;

*/	
	

for (int v=0; v<listVideos.size(); v++){
        vidFile=listVideos.at(v);
	cout<<endl<<"Video number from the list: "<<v+1<<endl<<vidFile<<endl;

        VideoCapture capture(vidFile);
        if(!capture.isOpened()) {
                printf("Could not initialize capturing..\n");
                return -1;
        }
	
	
	frame_num = 0;

	while(true) {
		capture >> frame;
		if(frame.empty())
			break;
		if(frame_num == 0) {

			image.create(frame.size(), CV_8UC3);
	//		grey.create(frame.size(), CV_8UC1);
	//		prev_image.create(frame.size(), CV_8UC3);
	//		prev_grey.create(frame.size(), CV_8UC1);
			
			
	//		frame.copyTo(prev_image);
			
		//	tstart = time(0);
	//		cvtColor(prev_image, prev_grey, CV_BGR2GRAY); //!!!!!!!Warning cvtColor for the first time is very time consuming
		//	tend = time(0);
                  //      cout <<endl<< "It took for cvtColor(prev_image, prev_grey, CV_BGR2GRAY);: "<< difftime(tend, tstart) <<" second(s)."<< endl;
			
			frame_num++;

			int step_t = step;
			while (step_t > 1){
				capture >> frame;
				step_t--;
			}
			continue;

		}

		frame.copyTo(image);
/*		cvtColor(image, grey, CV_BGR2GRAY);


		frame_0.upload(prev_grey);
		frame_1.upload(grey);


        // GPU optical flow
		switch(type){
		case 0:
			//alg_farn->calc(frame_0,frame_1,flow_u);
			//alg_farn->calc(frame_0,frame_1,flow_u,flow_v);
			alg_farn->calc(frame_0,frame_1,d_flow);
			break;
		case 1:
			//alg_tvl1(frame_0,frame_1,flow_u,flow_v);
			//alg_tvl1(frame_0,frame_1,flow_u);
			alg_tvl1->calc(frame_0,frame_1,d_flow);

			break;
		case 2:
			GpuMat d_frame0f, d_frame1f;
	        frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
	        frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
			//alg_brox(d_frame0f, d_frame1f, flow_u,flow_v);
			//alg_brox(d_frame0f, d_frame1f, flow_u);
			alg_brox->calc(d_frame0f, d_frame1f, d_flow);
			break;
		}

		//flow_u.download(flow_x);
		//flow_v.download(flow_y);
		GpuMat planes[2];
    		cuda::split(d_flow, planes);
		Mat flow_x(planes[0]);
    		Mat flow_y(planes[1]);

		// Output optical flow
		Mat imgX(flow_x.size(),CV_8UC1);
		Mat imgY(flow_y.size(),CV_8UC1);
		convertFlowToImage(flow_x,flow_y, imgX, imgY, -bound, bound);
*/		char tmp[20];
		sprintf(tmp,"_%04d.jpg",int(frame_num));

		Mat imgX_, imgY_, image_;
//		resize(imgX,imgX_,cv::Size(340,256));
//		resize(imgY,imgY_,cv::Size(340,256));
		resize(image,image_,cv::Size(340,256));
		
		
		pathSaveFrame = listRootSave.at(v) + saveFrame + tmp;
		
	/*	
		//c=pathSaveFrame.c_str();
		auto dir = opendir(pathSaveFrame.c_str());
		if(!dir){
			cout<<endl<<"Worning!!!!!!!!!! the directory: "<<pathSaveFrame<<" doesn't exit"<<endl;
		}
	*/			
		
		imwrite(pathSaveFrame, image_);
//		cout<<endl<<pathSaveFrame<<endl;

		//pathSaveFlow_x = listRootSave.at(v) + saveFlow_x + tmp;
		//imwrite(pathSaveFlow_x,imgX_);
//		cout<<endl<<pathSaveFlow_x<<endl;

		//pathSaveFlow_y = listRootSave.at(v) + saveFlow_y + tmp; 
		//imwrite(pathSaveFlow_y,imgY_);
//		cout<<endl<<pathSaveFlow_y<<endl;

	//	std::swap(prev_grey, grey);
	//	std::swap(prev_image, image);
		frame_num = frame_num + 1;

		int step_t = step;
		while (step_t > 1){
			capture >> frame;
			step_t--;
		}
	}
}
      
	return 0;
}
