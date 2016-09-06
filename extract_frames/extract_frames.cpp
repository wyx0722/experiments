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
 
  cout<<"OK"<<endl;
  string pathVideos=argv[1];
  string s_argv2 = argv[2];
  ifstream pathListVideos (s_argv2);
  string pathSaveFrames=argv[3];
  
  cout<<"OK"<<endl;
  
  //cout<<"pathVideos: "<<pathVideos<<"  pathListVideos: "<<pathListVideos<<"  pathSaveFrames "<<pathSaveFrames<<endl;
  
  string line, str;
  vector<string> listVideos;
  vector<string> fullPath_listVideos;
  
  
  
  
  if (pathListVideos.is_open())
  {
    while(getline(pathListVideos, line)){
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
  }
  else cout<<endl<<"Unable to open the file! \n";
  
  for (int i=0; i<fullPath_listVideos.size(); i++)
    cout<<i+1<<" "<<fullPath_listVideos.at(i)<<endl;
  
  
  //for (int i=0; i  )
  //string str=pathListVideos.substr(pathListVideos.length()-4, listVideos.length()-1);
  
  //string s_avi=".avi";
  
  
  //cout<<pathListVideos.length()<<endl<<str<<endl;
  
  cout<<"OK!!"<<endl;

}
