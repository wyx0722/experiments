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
  std::string listVideos=argv[2];
  string pathSaveFrames=argv[3];
  
  
  cout<<"pathVideos: "<<pathVideos<<"  listVideos: "<<listVideos<<"  pathSaveFrames "<<pathSaveFrames<<endl;
  
  string str=listVideos.substr(listVideos.length()-4, listVideos.length()-1);
  
  if (strcmp(str, ".avi")!=0)
  {
    cout<<"Does not ends with .avi"<<endl;
  }
  else{
    cout<<"ends with .avi"<<endl;
  }
  
  cout<<listVideos.length()<<endl<<str<<endl;
  
  cout<<"OK!!"<<endl;

}
