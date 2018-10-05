#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <assert.h>

#include <LBP.hpp>
#include <ProgressBar.hpp>
#include <Util.hpp>
#include <main.hpp>


int main(int argc, char** argv)
{
	std::string home = string("..");
	std::string dataDir = home+"/dataset";
	std::string trainDir = dataDir+"/train/";
	std::string testDir = dataDir+"/test/";

	std::vector<std::string> fileList = std::vector<std::string>();
	
	// prepare train data
	getFileList(trainDir, fileList);
	std::cout << fileList.size() << std::endl;
	cv::Mat train_data, train_label;
	DataPreparation(trainDir, fileList, train_data, train_label);

	/*
    for (unsigned int i = 0;i < list.size();i++) {
        std::cout << list[i] << std::endl;
    }
    */
    return 0;
}


void DataPreparation(const std::string dirPath, const std::vector<std::string>& filelist, cv::Mat& data, cv::Mat& label)
{
	if(filelist.size() == 0)
	{
		std::cout<< "ERROR: File list is empty. Data preparation aborted." << std::endl;
	}
	else
	{
		std::vector<cv::Mat> data_vector;
		std::vector<cv::Mat> label_vector;
		cv::Mat srcImg;
		cv::Mat resizedImg;


		LBP lbp_helper(cv::Size(cell_size, cell_size));
		int total_ticks = filelist.size();
		ProgressBar progressBar(total_ticks, 70, '=', '-');

		for(std::vector<std::string>::const_iterator it = filelist.begin(); it != filelist.end(); ++it)
		{
			srcImg = cv::imread(dirPath+*it, cv::IMREAD_COLOR);
			cv::resize(srcImg, resizedImg, cv::Size(resize_row, resize_col));
			cv::Mat sample_hist_vector;

			/* prepare data feature vector */
			// hsv
			cv::Mat hsv_image;
			cv::cvtColor(resizedImg, hsv_image, cv::COLOR_RGB2HSV);
			const std::vector<cv::Mat> hsv_channels = splitChannels(hsv_image);
			for(std::vector<cv::Mat>::const_iterator channel = hsv_channels.begin(); channel != hsv_channels.end(); ++channel)
			{
				cv::Mat channel_lbp_hist;
				lbp_helper.computeLBPFeatureVector_RI_Uniform(*channel, channel_lbp_hist);
				//if(hsv_lbp_hist.cols == 0) hsv_lbp_hist = channel_lbp_hist;
				//else lbp_helper.mergeCols(hsv_lbp_hist, channel_lbp_hist);
				sample_hist_vector = lbp_helper.mergeCols(sample_hist_vector, channel_lbp_hist);
			}

			// ycbcr
			cv::Mat ycbcr_image;
			cv::cvtColor(resizedImg, ycbcr_image, cv::COLOR_RGB2YCrCb);
			const std::vector<cv::Mat> ycbcr_channels = splitChannels(ycbcr_image);
			for(std::vector<cv::Mat>::const_iterator channel = ycbcr_channels.begin(); channel != ycbcr_channels.end(); ++channel)
			{
				cv::Mat channel_lbp_hist;
				lbp_helper.computeLBPFeatureVector_RI_Uniform(*channel, channel_lbp_hist);
				sample_hist_vector = lbp_helper.mergeCols(sample_hist_vector, channel_lbp_hist);
			}
			// push back sample
			data.push_back(sample_hist_vector);

			/* prepare label feature vector */
			if((*it).find("fake") != string::npos)  // real person
			{
				label.push_back(0);
			}
			else if((*it).find("living") != string::npos)  //fake attack
			{
				label.push_back(1);
			}

			++progressBar;
			progressBar.display();
		}
		progressBar.done();
		std::cout<<"feature vector rows: "<<data.rows<<"feature vector cols: "<<data.cols<<std::endl;
		std::cout<<"label vector rows: "<<label.rows<<"label vector cols: "<<label.cols <<std::endl;

		cv::FileStorage fs;
		fs.open("train_feature.xml", cv::FileStorage::WRITE);
		fs << "TrainFeature" << data;
		fs.release();
		fs.open("train_label.xml", cv::FileStorage::WRITE);
		fs << "TrainLabel" << label;
		fs.release();
	}
}