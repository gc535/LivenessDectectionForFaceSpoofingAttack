#include <opencv2/opencv.hpp>

#include <Util.hpp>
#include <LBP.hpp>
#include <Data.hpp>
#include <ProgressBar.hpp>
#include <OFM.hpp>

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>


int main(int argc, char** argv)
{
	std::cout<<"[Note]: This program generates the data file for from the optical flow images"<<std::endl;

	std::string data_path = "";
	int resize=0, cell_size=0;
	ParseArgument(argc, argv, resize, cell_size, data_path);

	std::cout<<data_path<<resize<<cell_size<<std::endl;
	std::string trainDir = data_path+"/train/";
	std::string testDir = data_path+"/test/";

	std::cout<<"[Note]: Starting data preparation phase..."<<std::endl;
	// prepare train data
	Data data(trainDir, Data::Action::TRAIN);
	cv::Mat train_data, train_label;
	data.DataPreparation(OFM_LBP, train_data, train_label, "ofm_lbp", resize, cell_size);
	
	// prepare test data
	data.update(testDir, Data::Action::TEST);
	cv::Mat test_data, test_label;
	data.DataPreparation(OFM_LBP, test_data, test_label, "ofm_lbp", resize, cell_size);

	return 0;
}

void OFM_LBP(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, int resize, int cell_size)
{
	cv::Mat srcImg, resizedImg; 							// source images from file
	//LBP lbp_face_cell(cv::Size(_cellsize, _cellsize));  	// create LBP extractor for face
	//LBP lbp_scene_cell(cv::Size(_cellsize, _cellsize));		// create LBP extractor for scene
	
	LBP lbp(cv::Size(cell_size, cell_size));

	int total_ticks = filelist.size();						// prepare progress bar
	ProgressBar progressBar(total_ticks, 70, '=', '-');		// prepare progress bar
	for(std::vector<std::string>::const_iterator it = filelist.begin(); it != filelist.end(); ++it)
	{
   		srcImg = cv::imread(*it, cv::IMREAD_GRAYSCALE);
		cv::resize(srcImg, resizedImg, cv::Size(resize, resize));

		cv::Mat sample_hist_vector;  // new container stores the complete feature vector for every sample
		lbp.computeLBPFeatureVector(resizedImg, sample_hist_vector, LBP::Mode::RIU2);
		
		data.push_back(sample_hist_vector);
		// /* prepare label feature vector 
		if((*it).find("fake") != string::npos)  
		{
			label.push_back(0);
		}
		else if((*it).find("living") != string::npos)  
		{
			label.push_back(1);
		}
		++progressBar;
		progressBar.display();
	}
	progressBar.done();  // terminate progress bar display

}

void ParseArgument(const int& argc, const char* const* argv,  
				   int& resize, int& cell_size,
				   std::string& data_path)
{
	if( argc < 7 || argc > 7) {
		printHelp();
		exit( 1 );
	}
	else if( argc == 7 ) {
		// process arguments
		for( int i = 1; i < argc - 1; i++ ) {
			if( strcmp( argv[i], "-d" ) == 0 ) {
				data_path = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-r" ) == 0 ) {
				resize = atoi( argv[i + 1] );
				i++;
			}
			else if( strcmp( argv[i], "-c" ) == 0 ) {
				cell_size = atoi( argv[i + 1] );
				i++;
			}
			else {
				std::cerr << "invalid argument: \'" << argv[i] << "\'\n";
				printHelp();
				exit( 1 );
			}
		}
	}
}

void printHelp() {
	std::cout << "\nUsage: ./combine [options]" << std::endl;
	std::cout << "\nOptions:" << std::endl;
	std::cout << "\t-r  <int> - Target resize size (default=96)" << std::endl;
	std::cout << "\t-c  <int> - Desired cell size for LBP extraction (default=16)" << std::endl;
	std::cout << "\t-d  <string> - path to iamge folder" << std::endl;
	std::cout << "\t \t \tData root should contain a two sub folder: 'train' and 'test" << std::endl;
}