#ifndef FFT_DOG_HPP
#define FFT_DOG_HPP
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <Data.hpp>
#include <ProgressBar.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui/highgui.hpp>

void test(cv::Mat test_data, cv::Mat test_label, cv::Ptr<cv::ml::SVM> svm);

void getFeature(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, 
		        const int resize, const int cellsize, const double sigma1, const double sigma2);

void findFrequencyReponse(cv::Mat& resizedImg, cv::Mat& response, const double sigma1, const double sigma2);

cv::Mat FFTDOG(cv::Mat srcImg, const double sigma1, const double sigma2);

void parseArguments(const int argc, const char* const* argv,
					int& resize, int& cellsize, std::string& train_list, std::string& test_list);

void printHelp();

#endif //FFT_DOG_HPP