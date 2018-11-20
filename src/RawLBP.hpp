#pragma once

#include <opencv2/opencv.hpp>

#include <Util.hpp>
#include <Data.hpp>
#include <ProgressBar.hpp>

#include <string>
#include <vector>

void RawLBP(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, 
            const int resize, const int cellsize);

void parseArguments(const int argc, const char* const* argv,
                    int& resize, int& cellsize, std::string& train_list, std::string& test_list);

void printHelp();