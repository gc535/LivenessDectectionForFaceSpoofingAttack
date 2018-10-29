#include <Util.hpp>
#include <LBP.hpp>
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
	std::string trainFaceDir = data_path+"/train/face";
	std::string trainSceneDir = data_path+"/train/scene";
	std::string testFaceDir = data_path+"/test/scene";
	std::string testSceneDir = data_path+"/test/face";


	/*
	std::vector<std::string> trainFaces;
	std::vector<std::string> testFaces;
	if(getFilelist(trainFaceDir, trainFaces))
	{
		std::cout<<"[Error]: Failed to get the image filelist. Exiting..."<<std::endl;
		exit(1);
	} 
	if(getFilelist(testFaceDir, testFaces))
	{
		std::cout<<"[Error]: Failed to get the image filelist. Exiting..."<<std::endl;
		exit(1);
	}

	cv::Mat srcImg, resizedImg; 							// source images from file
	LBP lbp_face_cell(cv::Size(_cellsize, _cellsize));  	// create LBP extractor for face
	LBP lbp_scene_cell(cv::Size(_cellsize, _cellsize));		// create LBP extractor for scene
	cv::Mat data, label;  									// data and label matrix to be prepared
	int total_ticks = filelist.size();						// prepare progress bar
	ProgressBar progressBar(total_ticks, 70, '=', '-');		// prepare progress bar
	for(std::vector<std::string>::iterator it = filelist.begin(); it != filelist.end(); ++it)
	{
   		srcImg = cv::imread(_dirPath+*it, cv::IMREAD_GRAYSCALE);
		cv::resize(srcImg, resizedImg, cv::Size(resize, resize));
		int blocklen = resizedImg.rows/3;
		cv::Mat center_face = resizedImg(cv::Range(blocklen, 2*blocklen), cv::Range(blocklen, 2*blocklen)); // center block in all 9 blocks

		cv::Mat sample_hist_vector;  // new container stores concatenated LBP feature vectors of a single sample
		// face
		cv::Mat face_lbp_hist_cell;
		lbp_cell.computeLBPFeatureVector(center_face, face_lbp_hist_cell, LBP::Mode::RIU2);
		sample_hist_vector = mergeCols(sample_hist_vector, face_lbp_hist_cell);

		//scene
		cv::Mat scene_lbp_hist_cell;
		lbp_full.computeLBPFeatureVector(resizedImg, scene_lbp_hist_cell, LBP::Mode::RIU2);
		sample_hist_vector = mergeCols(sample_hist_vector, scene_lbp_hist_cell);
		
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
	saveMatToHDF5(data.clone(), label.clone(), action_name);
	*/


	return 0;
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