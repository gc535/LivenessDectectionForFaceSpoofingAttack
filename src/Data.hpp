#ifndef DATA_HPP
#define DATA_HPP

#include <opencv2/opencv.hpp>

#include <Util.hpp>

#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>


class Data
{
public: 
	enum Action
	{
		TRAIN,
		TEST
	};

private:
	std::string _dirPath;
	Action _action;
	std::vector<std::string> _filelist;


public: 
	Data(){}
	Data(const std::string path, Action action)
	{
		_dirPath = path; 
		_action = action;
		updateFileList();
	}

	/* template function for data preperation */
	/* @input1: function pointer to a specified data preparation routine.
	   @input2: data matrix reference
	   @input3: label matrix reference
	   @input4: output file name
	   @other input[s]: inputs specific for different variadic functions
	   @return: void 
	*/
	template<typename ReturnType, typename MATdata, typename MATlabel, typename name_string, typename... Args>
	void DataPreparation(ReturnType (*method)(MATdata&, MATlabel&, const std::vector<std::string>&, Args...),  
						 MATdata& data, MATlabel& label, const name_string output_file_prefix, Args... args)
	{
		if(_filelist.size() == 0)
		{
			std::cout<< "[ERROR]: File list is empty. Data preparation aborted." << std::endl;
			exit(1);
		}
		else
		{   
			/* construct output file name based on prefix and action */
			const std::string action_name = (_action==TRAIN) ? "train" : "test";
			std::cout<<"[NOTE]: Preparing "<<action_name<<" data..."<<std::endl;
			std::string data_name = (_action==TRAIN) ? "TrainFeature" : "TestFeature";
			std::string label_name = (_action==TRAIN) ? "TrainLabel" : "TestLabel";
			std::string output_file_name = output_file_prefix+std::string("_")+action_name;       // taget output file name

			if(!exists(output_file_name+".h5"))
			{
				method(data, label, _filelist, args...);

				std::cout<<"[Note]:feature vector rows: "<<data.rows<<"  feature vector cols: "<<data.cols<<std::endl;
	 			std::cout<<"[Note]:label vector rows: "<<label.rows<<"  label vector cols: "<<label.cols <<std::endl;

	 			std::cout<<"[Note]: Saving data files..."<<std::endl;
	 			//save to txt for python 
	 			saveMatToHDF5(data.clone(), label.clone(), output_file_name);
			}
			else
			{
				std::cout<<"[Note]: Target database file found in the directory, skipping..."<<std::endl;
			}
			
		}
	}

	//void DataPreparation(void (*method)(cv::Mat&, cv::Mat&, const std::vector<std::string>& ), cv::Mat& data, cv::Mat& label);
	
	void changePath(const std::string path){_dirPath = path;}
	void changeAction(Action action){_action = action;}
	
	void update(const std::string path, Action action);
	

private:
	/* updata the _filelist after update the directory */
	int updateFileList();
	/* get length of _filelist */
	int getNumOfSamples();
	/* check if file exists */
	bool exists (const std::string& name);


};


#endif //DATA_HPP
