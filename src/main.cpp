#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>



#include <main.hpp>
#include <Data.hpp>
#include <ProgressBar.hpp>

void test(cv::Mat, cv::Mat, cv::Ptr<cv::ml::SVM>);

int main(int argc, char** argv)
{
	int resize = 96;
	int cellsize = 16;
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

	
	// prepare train data
	Data data(trainDir, Data::Action::TRAIN, resize, cellsize);
	cv::Mat train_data, train_label;
	data.DataPreparation(train_data, train_label);

	
	data.update(testDir, Data::Action::TEST, resize, cellsize);
	cv::Mat test_data, test_label;
	data.DataPreparation(test_data, test_label);

	// training 

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  	svm->setType(cv::ml::SVM::Types::C_SVC);
  	svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
  	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER,100,1e-6));
  	std::cout << "[Note]: Training SVM......" << std::endl;
  	try 
  	{
		//svm = StatModel::train<SVM>(train_feature, ROW_SAMPLE, train_label, params);//train SVM
		svm->train(train_data,cv::ml::SampleTypes::ROW_SAMPLE,train_label);
  		std::cout << "[Note]: Training finished......" << std::endl;
  		svm->save("liveness_svm.xml");
  	} 
  	catch (cv::Exception& e) 
  	{
		std::cout << e.msg;
  	}

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

