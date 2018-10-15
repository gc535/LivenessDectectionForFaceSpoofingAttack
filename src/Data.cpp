// core dependency
#include <opencv2/opencv.hpp>

// local include
#include <LBP.hpp>
#include <ProgressBar.hpp>
#include <Util.hpp>
#include <Data.hpp>
#include <SingleChannelDOG.hpp>

// util
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>
#include <fstream>

int Data::updateFileList()
{
    DIR *dp;
    struct dirent *filePath;
    if((dp  = opendir(_dirPath.c_str())) == NULL) {
        std::cout << "[ERROR]: Error(" << errno << ") opening " << _dirPath << std::endl;
        return errno;
    }

    _filelist.clear(); //clear old file list
    struct stat path_stat;  // checking path type
    while ((filePath = readdir(dp)) != NULL) 
    {
    	stat((_dirPath+string(filePath->d_name)).c_str(), &path_stat);
    	if(S_ISREG(path_stat.st_mode))   // is file
    	{
    		_filelist.push_back(string(filePath->d_name));
    	}
    }
    closedir(dp);
    return 0;
}

void Data::update(const std::string path, Action action, int resize, int cellsize)
{
	_dirPath = path; 
	_action = action;
	_resize = resize;
	_cellsize = cellsize;
	updateFileList();
}

void Data::DataPreparation(cv::Mat& data, cv::Mat& label)
{
	if(_filelist.size() == 0)
	{
		std::cout<< "[ERROR]: File list is empty. Data preparation aborted." << std::endl;
		exit(1);
	}
	else
	{   
		const string action_name = (_action==TRAIN) ? "train" : "test";
		std::cout<<"[NOTE]: Preparing "<<action_name<<" data..."<<std::endl;
		std::string data_name = (_action==TRAIN) ? "TrainFeature" : "TestFeature";
		std::string label_name = (_action==TRAIN) ? "TrainLabel" : "TestLabel";
		
		if (!exists(data_name+".xml") || !exists(label_name+".xml"))
		{
			std::vector<cv::Mat> data_vector;
			std::vector<cv::Mat> label_vector;
			cv::Mat srcImg;
			cv::Mat resizedImg;


			LBP lbp_cell(cv::Size(_cellsize, _cellsize));
			LBP lbp_full(cv::Size(_resize, _resize));
			int total_ticks = _filelist.size();
			ProgressBar progressBar(total_ticks, 70, '=', '-');

			for(std::vector<std::string>::iterator it = _filelist.begin(); it != _filelist.end(); ++it)
			{
				srcImg = cv::imread(_dirPath+*it, cv::IMREAD_COLOR);
				cv::resize(srcImg, resizedImg, cv::Size(_resize, _resize));
				

				/* prepare feature vector for every sample*/
				std::vector<cv::Mat> dog_channels;
				const std::vector<cv::Vec2d> vector_sigmas = { cv::Vec2d(0.5, 1), cv::Vec2d(1, 2), cv::Vec2d(0.5,2)};
				
				
				// hsv
				cv::Mat hsv_image;
				cv::cvtColor(resizedImg, hsv_image, cv::COLOR_RGB2HSV);
				const std::vector<cv::Mat> hsv_channels = splitChannels(hsv_image);
				for(std::vector<cv::Mat>::const_iterator channel = hsv_channels.begin(); channel != hsv_channels.end(); ++channel)
				{
					SingalChannelImageDoG(*channel, vector_sigmas, dog_channels);
				}
				
				
				// ycbcr
				cv::Mat ycbcr_image;
				cv::cvtColor(resizedImg, ycbcr_image, cv::COLOR_RGB2YCrCb);
				const std::vector<cv::Mat> ycbcr_channels = splitChannels(ycbcr_image);
				for(std::vector<cv::Mat>::const_iterator channel = ycbcr_channels.begin(); channel != ycbcr_channels.end(); ++channel)
				{
					SingalChannelImageDoG(*channel, vector_sigmas, dog_channels);
				}
				
				// LBP on all DOG image channels
				//std::cout<<"dog channels nums: "<<dog_channels.size()<<std::endl;
				cv::Mat sample_hist_vector;  
				for(std::vector<cv::Mat>::iterator channel = dog_channels.begin(); channel != dog_channels.end(); ++channel)
				{
					//std::cout<<"rows: "<<sample_hist_vector.rows<<"cols: "<<sample_hist_vector.cols<<std::endl;
					cv::Mat channel_lbp_hist_cell;
					lbp_cell.computeLBPFeatureVector(*channel, channel_lbp_hist_cell, LBP::Mode::RIU2);
					sample_hist_vector = mergeCols(sample_hist_vector, channel_lbp_hist_cell);
					cv::Mat channel_lbp_hist_full;
					lbp_full.computeLBPFeatureVector(*channel, channel_lbp_hist_full, LBP::Mode::RIU2);
					sample_hist_vector = mergeCols(sample_hist_vector, channel_lbp_hist_full);
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
			std::cout<<"[Note]:feature vector rows: "<<data.rows<<"  feature vector cols: "<<data.cols<<std::endl;
			std::cout<<"[Note]:label vector rows: "<<label.rows<<"  label vector cols: "<<label.cols <<std::endl;

			std::cout<<"[Note]: Saving data files..."<<std::endl;
			//save to txt for python 
			writeMatToFile(data,data_name+".txt");
			writeMatToFile(label,label_name+".txt");

			// save to xml
			cv::FileStorage fs;
			fs.open(data_name+".xml", cv::FileStorage::WRITE);
			fs << data_name << data;
			fs.release();
			fs.open(label_name+".xml", cv::FileStorage::WRITE);
			fs << label_name << label;
			fs.release();
		}
		else
		{
			std::cout<<"[Note]: Data found in the directory, load it from file..."<<std::endl;
			cv::FileStorage fs;
			fs.open(data_name+".xml", cv::FileStorage::READ);
			fs[data_name] >> data;
			fs.release();

			fs.open(label_name+".xml", cv::FileStorage::READ);
			fs[label_name] >> label;
			fs.release();
			std::cout<<"[Note]: load complete!"<<std::endl;
		}
		
	}
}

int Data::getNumOfSamples()
{
	return _filelist.size();
}


/* check if file exists */
bool Data::exists (const std::string& name) 
{
    return ( access( name.c_str(), F_OK ) != -1 );
}