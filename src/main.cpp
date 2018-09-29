#include <stdlib.h>
#include <vector>
#include <string>
#include <Util.hpp>
#include "opencv2/opencv.hpp"
#include <LBP.hpp>
#include <iostream>



int main(int argc, char** argv)
{
	std::string home = string("..");
	std::string data = home+"/dataset";
	std::string test = data+"/train/";

	std::vector<std::string> list = std::vector<std::string>();
	getFileList(test, list);
	std::cout << list.size() << std::endl;
	/*
    for (unsigned int i = 0;i < list.size();i++) {
        std::cout << list[i] << std::endl;
    }
    */
    return 0;
}
