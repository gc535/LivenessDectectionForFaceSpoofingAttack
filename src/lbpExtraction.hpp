#ifndef LBP_EXTRACTION_HPP
#define LBP_EXTRACTION_HPP

#include <stdlib.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

enum Action
{
	TRAIN,
	TEST
};


void printHelp();
void DOG_LBP(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, int resize, int cellsize);

#endif //LBP_EXTRACTION_HPP