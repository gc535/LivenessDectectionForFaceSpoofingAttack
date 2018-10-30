#ifndef OFM_HPP
#define OFM_HPP

#include <opencv2/opencv.hpp>

#include <stdlib.h>
#include <vector>
#include <string>

void OFM_LBP(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, int resize, int cell_size);
void ParseArgument(const int& argc, const char* const* argv,  
				   int& resize, int& cell_size,
				   std::string& data_path);
void printHelp();


#endif //OFM_HPP