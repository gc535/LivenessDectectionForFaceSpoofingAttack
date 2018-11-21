#pragma once

#include <lbp_hist.hpp>
#include <lbpExtraction.hpp>
#include <Util.hpp>
#include <ProgressBar.hpp>
#include <Data.hpp>
#include <LBP.hpp>
#include <SingleChannelDOG.hpp>


void combineFeatures(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, int resize, int cellsize);

void DOG_LBP(cv::Mat& srcImg, cv::Mat& sample_hist_vector, const std::vector<cv::Vec2d> vector_sigmas, const int resize, const int cellsize);


void parseArguments(const int argc, const char* const* argv,
                    int& resize, int& cellsize, std::string& train_list, std::string& test_list);
void printHelp();