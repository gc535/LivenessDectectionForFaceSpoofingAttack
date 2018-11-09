// core dependency
#include <opencv2/opencv.hpp>

// local include
//#include <lbpExtraction.hpp>
#include <Util.hpp>
#include <Data.hpp>

// util
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>
#include <fstream>

int Data::updateFileList()
{   
    /* this method is updated with filelist reading
    DIR *dp;
    struct dirent *filePath;
    if((dp  = opendir(_dirPath.c_str())) == NULL) {
        std::cout << "[ERROR]: Error(" << errno << ") opening " << _dirPath << std::endl;
        return errno;
    }

    _filelist.clear(); //clear old file list
    struct stat path_stat;  // checking path type
    while ((filePath = readdir(dp)) != NULL) 
    {
    	stat((_dirPath+std::string(filePath->d_name)).c_str(), &path_stat);
    	if(S_ISREG(path_stat.st_mode))   // is file
    	{
    		_filelist.push_back(_dirPath+std::string(filePath->d_name));
    	}
    }
    closedir(dp);
    */
    if(!exists(_filelist_path)){
        std::cout << "[ERROR]: Error(" << errno << ") opening " << _filelist_path << std::endl;
        return errno;
    }

    std::ifstream files(_filelist_path);
    _filelist.clear(); //clear old file list
    std::string file;
    while(files >> file ){
        _filelist.push_back(file);
    }
    return 0;
}

void Data::update(const std::string filelist_path, Action action)
{
	_filelist_path = filelist_path; 
	_action = action;
	updateFileList();
}


int Data::getNumOfSamples()
{
	return _filelist.size();
}


/* check if file exists */
bool Data::exists (const std::string& name) 
{
    return ( access( name.c_str(), F_OK ) != -1 );
}