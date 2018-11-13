#ifndef FFT_DOG_HPP
#define FFT_DOG_HPP
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>

#include <Data.hpp>
#include <ProgressBar.hpp>

#include <opencv2/opencv.hpp>


void FFTDOG(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, 
		    const int resize, const int cellsize, const double sigma1, const double sigma2);

cv::Mat single_FFTDOG(cv::Mat srcImg, double sigma1, double sigma2);

void parseArguments(const int argc, const char* const* argv,
					int& resize, int& cellsize, std::string& train_list, std::string& test_list);

void printHelp();

#endif //FFT_DOG_HPP