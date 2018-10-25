#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>

#include <combine.hpp>
#include <ProgressBar.hpp>
#include <Util.hpp>


#include <stdlib.h>
#include <string>
#include <iostream>


int main(int argc, char** argv)
{

	std::string model_1 = "";
	std::string checkpoint_1 = "";
	std::string model_2 = "";
	std::string checkpoint_2 = "";
	std::string data_path = "";
	int resize = 0, cell_size = 0;
	ParseArgument(argc, argv, model_1, checkpoint_1, model_2, 
				  checkpoint_2, resize, cell_size, data_path);

	cv::dnn::Net net;
	/*
    try {
        net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
    }
    catch (cv::Exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        if (net.empty())
        {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "prototxt:   " << modelTxt << std::endl;
            std::cerr << "caffemodel: " << modelBin << std::endl;
            std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
            std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
            exit(-1);
        }
    }
    if (parser.get<bool>("opencl"))
    {
        net.setPreferableTarget(DNN_TARGET_OPENCL);
    }
    Mat img = imread(imageFile);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }
    //GoogLeNet accepts only 224x224 BGR-images
    Mat inputBlob = blobFromImage(img, 1.0f, Size(224, 224),
                                  Scalar(104, 117, 123), false);   //Convert Mat to batch of images
    net.setInput(inputBlob, "data");        //set the network input
    Mat prob = net.forward("prob");         //compute output
    cv::TickMeter t;
    for (int i = 0; i < 10; i++)
    {
        CV_TRACE_REGION("forward");
        net.setInput(inputBlob, "data");        //set the network input
        t.start();
        prob = net.forward("prob");                          //compute output
        t.stop();
    }
    int classId;
    double classProb;
    getMaxClass(prob, &classId, &classProb);//find the best class
    std::vector<String> classNames = readClassNames();
    std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << std::endl;
    */

    return 0;
}


void ParseArgument(const int& argc, const char* const* argv, std::string& model_1, std::string& checkpoint_1, 
				  std::string& model_2, std::string& checkpoint_2, int& resize, int& cell_size,
				  std::string& data_path)
{
	if( argc < 11 ) {
		printHelp();
		exit( 1 );
	}
	else if( argc >= 11 ) {
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
			else if( strcmp( argv[i], "-m1" ) == 0 ) {
				model_1 = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-p1" ) == 0 ) {
				checkpoint_1 = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-m2" ) == 0 ) {
				model_2 = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-p2" ) == 0 ) {
				checkpoint_2 = argv[i + 1];
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
	std::cout << "\t-m1 <string> - LBP model path (*.prototxt)" << std::endl;
	std::cout << "\t-p1 <string> - LBP model parameters (*.caffemodel)" << std::endl;
	std::cout << "\t-m2 <string> - OFM model path (*.prototxt)" << std::endl;
	std::cout << "\t-p2 <string> - OFM model parameters (*.caffemodel)" << std::endl;
	std::cout << "\t-r  <int> - Target resize size (default=96)" << std::endl;
	std::cout << "\t-c  <int> - Desired cell size for LBP extraction (default=16)" << std::endl;
	std::cout << "\t-d  <string> - path to iamge folder" << std::endl;
	std::cout << "\t \t \tData root should contain a two sub folder: 'train' and 'test" << std::endl;
}