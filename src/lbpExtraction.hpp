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
void parseArguments(const int& argc, const char* const* argv,
					int& resize, int& cellsize, std::string& train_list, std::string& test_list, Action& action);
void getFeature(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, int resize, int cellsize);
void DOG_LBP(cv::Mat& srcImg, cv::Mat& sample_hist_vector, const std::vector<cv::Vec2d> vector_sigmas, const int resize, const int cellsize); 
void test(cv::Mat, cv::Mat, cv::Ptr<cv::ml::SVM>);

#endif //LBP_EXTRACTION_HPP