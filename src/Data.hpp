#ifndef DATA_HPP
#define DATA_HPP

//#include <stdlib.h>
#include <string>
#include <vector>


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
	Data(const std::string path, Action action)
	{
		_dirPath = path; 
		_action = action;
		updateFileList();
	}


	void DataPreparation(cv::Mat& data, cv::Mat& label);
	
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