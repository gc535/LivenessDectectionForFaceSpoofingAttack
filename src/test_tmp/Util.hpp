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
    	stat(filePath->d_name, &path_stat);
    	if(S_ISREG(path_stat.st_mode))
    	{
    		files.push_back(string(filePath->d_name));
    	}
    }
    closedir(dp);
    return 0;
}


void ProgressBar(float precent)
{
    
    if (precent <= 1.0) {
        int barWidth = 70;

        std::cout << "[";
        int pos = barWidth * precent;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(precent * 100.0) << " %\r";
        std::cout.flush();

        //precent += 0.16; // for demonstration only
    }
    std::cout << std::endl;

}