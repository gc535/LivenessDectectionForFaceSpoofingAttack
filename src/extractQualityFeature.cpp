#include <extractQualityFeature.hpp>
#include <string>

int main(int argc, char** argv)
{   
    int resize = 64;
    int cellsize = 8;
    std::string train_list = "", test_list = "";
    parseArguments(argc, argv,
                   resize, cellsize, train_list, test_list);
    
    // prepare train data
    Data data(train_list, Data::Action::TRAIN);
    cv::Mat train_data, train_label;
    data.DataPreparation(extractQualityFeature, train_data, train_label, "quality_feature");

    // prepare test data
    data.update(test_list, Data::Action::TEST);
    cv::Mat test_data, test_label;
    data.DataPreparation(extractQualityFeature, test_data, test_label, "quality_feature");

    return 0;
}

void extractQualityFeature(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist)
{
    cv::Mat srcImg;
    cv::Mat resizedImg;
    std::vector<double> hist;
    

    int total_ticks = filelist.size();
    ProgressBar progressBar(total_ticks, 70, '=', '-');
    for(std::vector<std::string>::const_iterator it = filelist.begin(); it != filelist.end(); ++it)
    {
        cv::Mat feature;
        srcImg = cv::imread(*it, cv::IMREAD_COLOR);
        std::vector<double> featurevector;
        ComputeBrisqueFeature(srcImg, featurevector);
        feature.push_back(featurevector);
        
        data.push_back(feature);
        //std::cout << "sample_hist row: " << sample_hist.rows << ", cols: " << sample_hist.cols << std::endl;

        /* prepare label feature vector */
        if((*it).find("fake") != std::string::npos)  
        {
            label.push_back(0);
        }
        else if((*it).find("living") != std::string::npos)  
        {
            label.push_back(1);
        }

        ++progressBar;
        progressBar.display();
    }
    progressBar.done();

}

void parseArguments(const int argc, const char* const* argv,
                    int& resize, int& cellsize, std::string& train_list, std::string& test_list)
{
    if( argc <= 2 ) {
        printHelp();
        exit( 1 );
    }
    else if( argc > 2 ) {
        // process arguments
        for( int i = 1; i < argc - 1; i++ ) {
            if( strcmp( argv[i], "-t" ) == 0 ) {
                train_list = argv[i + 1];
                i++;
            }
            else if( strcmp( argv[i], "-v" ) == 0 ){
                test_list = argv[i + 1];
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
    if (train_list == "" or test_list == ""){
        printHelp();
        exit( 1 );
    }

}

void printHelp()
{
    std::cout << "\nUsage: ./lbp [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "\t-r  <int> - Target resize size (default=64)" << std::endl;
    std::cout << "\t-c  <int> - Desired cell size for LBP extraction (default=8)" << std::endl;
    std::cout << "\t-t <string> - Path to a txt file contains a list of training data" << std::endl;
    std::cout << "\t-v <string> - Path to a txt file contains a list of testing data" << std::endl;
}