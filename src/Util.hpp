#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>

using namespace std;

int getFileList (const string& dirPath, vector<string>& files)
{
    DIR *dp;
    struct dirent *filePath;
    if((dp  = opendir(dirPath.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dirPath << endl;
        return errno;
    }

    struct stat path_stat;
    while ((filePath = readdir(dp)) != NULL) 
    {
    	stat((dirPath+string(filePath->d_name)).c_str(), &path_stat);
    	if(S_ISREG(path_stat.st_mode))
    	{
    		files.push_back(string(filePath->d_name));
    	}
    }
    closedir(dp);
    return 0;
}

const std::vector<cv::Mat> splitChannels(const cv::Mat& MultiChannelImage)
{
    CV_Assert(MultiChannelImage.channels() > 0);
    std::vector<cv::Mat> splited_channels;
    cv::split(MultiChannelImage, splited_channels);
    return splited_channels;
} 