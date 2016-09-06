#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
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
using namespace std;

int main(int argc, char** argv){

  string pathVideos=argv[1];
  char* s_argv2 = argv[2];
  ifstream pathListVideos (argv[2]);
  //ifstream pathListVideos ("/home/ionut/experiments/extract_frames/tmp/list_v.txt");
  string pathSaveFrames=argv[3];
  
  
cout<<"pathVideos: "<<pathVideos<<"\npathListVideos: "<<s_argv2<<"\npathSaveFrames "<<pathSaveFrames<<endl<<endl;
  
  string line, str;
  vector<string> listVideos;
  vector<string> fullPath_listVideos;
  
  
  
  
  if (pathListVideos.is_open())
  {
    while(getline(pathListVideos, line)){
     if (!line.empty()){
	       //cout<<"Line: "<<line<<endl;
	       listVideos.push_back(line);
	 
	       str=line.substr(line.length()-4, line.length()-1);
	       if (str.compare(".avi")!=0)
	       {
	         fullPath_listVideos.push_back(pathVideos + line + ".avi");
	         //cout<<"Does not ends with .avi"<<endl;
	       }
	       else{
	         //cout<<"ends with .avi"<<endl;
	         fullPath_listVideos.push_back(pathVideos + line);
	       }
	     }
	     else cout<<"Line empty: "<<line<<endl;
    }
     pathListVideos.close();
  }
  else cout<<endl<<"Unable to open the file! \n";
  
  //cout<<"OK"<<endl;
  //for (int i=0; i<fullPath_listVideos.size(); i++)
    //cout<<i+1<<" "<<fullPath_listVideos.at(i)<<endl;
  
  
  DIR* dir;
  const char* c;
  string::size_type first=0;
  string classVideo, nameVideo, tmp;
  
  Mat frame;
  
  cout<<"Frame extraction for "<<fullPath_listVideos.size()<<" videos!\n\n";
  for (int i=0; i<fullPath_listVideos.size(); i++)
  {
  	cout<<i+1<<" "<<fullPath_listVideos.at(i)<<endl;
  	
  	VideoCapture capture(fullPath_listVideos.at(i));
  	if(!capture.isOpened()) {
                printf("Could not initialize capturing..\n");
                return -1;
        }
        
        first=listVideos.at(i).find("/");
        classVideo=listVideos.at(i).substr(0, first);
        nameVideo=listVideos.at(i).substr(first+1);
        //cout<<classVideo<<" "<<nameVideo<<endl;
        
        tmp=pathSaveFrames+classVideo;
        c=tmp.c_str();
        dir=opendir(c);
	if (!dir){
	        cout<<dir<<endl;;
	        mkdir(c, 0777);
	}
	
	tmp=pathSaveFrames+classVideo + "/" + nameVideo;
	c=tmp.c_str();
        dir=opendir(c);
        if (!dir){
                cout<<dir<<endl;
                mkdir(c, 0777);
        }
        
        int frame_num=1;
        while(true) {
		capture >> frame;
		if(frame.empty())
			break;
	
	char nameFrame[10];
	
	sprintf(nameFrame, "%06d.jpg", int(frame_num));
	
	string final_save_f=pathSaveFrames+classVideo + "/" + nameVideo+"/"+nameFrame;
	imwrite(final_save_f, frame);
	
	frame_num++;	
        }
        
        
  	
  }
  
  cout<<"Done!!"<<endl;

}
