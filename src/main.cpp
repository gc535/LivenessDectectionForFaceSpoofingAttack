#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include <main.hpp>
#include <Data.hpp>
#include <ProgressBar.hpp>
#include <Util.hpp>

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <errno.h>


//#include <Util.hpp>

void test(cv::Mat, cv::Mat, cv::Ptr<cv::ml::SVM>);

int main(int argc, char** argv)
{
	int resize = 96;
	int cellsize = 16;
	Action action = TRAIN;
	std::string dataDir = std::string("..");

	if( argc <= 2 ) {
		printHelp();
		exit( 1 );
	}
	else if( argc > 2 ) {
		// process arguments
		for( int i = 1; i < argc - 1; i++ ) {
			if( strcmp( argv[i], "-d" ) == 0 ) {
				dataDir = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-r" ) == 0 ) {
				resize = atoi( argv[i + 1] );
				i++;
			}
			else if( strcmp( argv[i], "-c" ) == 0 ) {
				cellsize = atoi( argv[i + 1] );
				i++;
			}
			else if( strcmp( argv[i], "-a" ) == 0 ) {
				if( strcmp( argv[i+1], "test" ) == 0 )
				{
					action = TEST;
				}
				else if( strcmp( argv[i+1], "train" ) == 0 )
				{
					action = TRAIN;
				}
				else
				{
					std::cout<<"[ERROR]: Invalid action. Action can only be 'train' or 'test'."<<std::endl;
					std::cout<<"\t\t\t Execution aborted."<<std::endl;
					exit( 1 );
				}
				i++;
			}
			else {
				std::cerr << "invalid argument: \'" << argv[i] << "\'\n";
				printHelp();
				exit( 1 );
			}
		}
	}
	std::string trainDir = dataDir+"/train/";
	std::string testDir = dataDir+"/test/";
	std::string tmpDir = dataDir+"/tmp/";

	std::cout<<"[Note]: Starting data preparation phase..."<<std::endl;
	// prepare train data
	Data data(trainDir, Data::Action::TRAIN, resize, cellsize);
	cv::Mat train_data, train_label;
	if(action != TEST)
	{
		data.DataPreparation(train_data, train_label);
	}
	
	// prepare test data
	data.update(testDir, Data::Action::TEST, resize, cellsize);
	cv::Mat test_data, test_label;
	data.DataPreparation(test_data, test_label);


	std::cout<<"[Note]: Starting model traning phase..."<<std::endl;
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	// training 
	if(!exists("liveness_svm.xml"))
	{
		svm->setType(cv::ml::SVM::Types::C_SVC);
	  	svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
	  	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER,100,1e-6));
	  	std::cout << "[Note]: Training SVM......" << std::endl;
	  	try 
	  	{
			//svm = StatModel::train<SVM>(train_feature, ROW_SAMPLE, train_label, params);//train SVM
			svm->train(train_data, cv::ml::SampleTypes::ROW_SAMPLE, train_label);
	  		std::cout << "[Note]: Training finished......" << std::endl;
	  		svm->save("liveness_svm.xml");
	  	} 
	  	catch (cv::Exception& e) 
	  	{
			std::cout << e.msg;
	  	}
	}
	else
	{
		std::cout << "[Note]: Pre-trained SVM model found in the current directory, load it from file......" << std::endl;
		svm = cv::Algorithm::load<cv::ml::SVM>("liveness_svm.xml");
	}
  	std::cout<<"[Note]: Model traning phase complete!"<<std::endl;


	std::cout<<"[Note]: Starting model testing phase..."<<std::endl;
  	std::cout<<"[Note]: Test set size: "<<test_data.rows<<" samples"<< std::endl;
  	test(test_data, test_label, svm);

    return 0;
}


void test(cv::Mat test_data, cv::Mat test_label, cv::Ptr<cv::ml::SVM> svm)
{
	CV_Assert(test_data.rows == test_label.rows && test_data.rows>0);

	std::cout << "[Note]: Start Testing......" << std::endl;
	int count = 0;
	int correct = 0;
	ProgressBar progressBar(test_data.rows, 70, '=', '-');
	for(int r = 0; r < test_data.rows; ++r)
	{
		int response = svm->predict(test_data.row(r));
		//std::cout<<"response: "<< response << ", expected: " <<  test_label.at<int>(r, 0) << std::endl;
		if (response == test_label.at<int>(r, 0)) correct++;
		count++;
		++progressBar;
		progressBar.display();
	}
	progressBar.done();
	std::cout<<"[Note]: Current model accuracy is: " << float(correct)/count*100 << "%" << std::endl;
}

