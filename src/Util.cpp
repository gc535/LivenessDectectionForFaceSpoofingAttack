#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <unistd.h>
#include <fstream>


cv::Mat mergeRows(const cv::Mat& A, const cv::Mat& B)
{
     int totalRows = A.rows + B.rows;
     cv::Mat mergedDescriptors(totalRows, A.cols, A.type());
     cv::Mat submat = mergedDescriptors.rowRange(0, A.rows);
     A.copyTo(submat);
     submat = mergedDescriptors.rowRange(A.rows, totalRows);
     B.copyTo(submat);
     return mergedDescriptors;
}

cv::Mat mergeCols(const cv::Mat& A, const cv::Mat& B)
{
    // If A and B both contains data, then their data type should be the same
    // or at least one of them contains dataa
    CV_Assert( (!A.empty() && !B.empty() && A.type()==B.type() && A.rows==B.rows)
                || (!A.empty() || !B.empty()) );

    if(!A.empty() && !B.empty())
    {
        int totalCols = A.cols + B.cols;
        cv::Mat mergedDescriptors(A.rows, totalCols, A.type());
        cv::Mat submat = mergedDescriptors.colRange(0, A.cols);
        A.copyTo(submat);
        submat = mergedDescriptors.colRange(A.cols, totalCols);
        B.copyTo(submat);
        return mergedDescriptors;
    }
    else if(!A.empty())
    {
        return A;
    }
    else
    {
        return B;
    }
}

const std::vector<cv::Mat> splitChannels(const cv::Mat& MultiChannelImage)
{
    CV_Assert(MultiChannelImage.channels() > 0);
    std::vector<cv::Mat> splited_channels;
    cv::split(MultiChannelImage, splited_channels);
    return splited_channels;
} 

/* check if file exists */
bool exists (const std::string& name) 
{
    return ( access( name.c_str(), F_OK ) != -1 );
}

// saving data to txt
void writeMatToFile(cv::Mat& m, const std::string& filename)
{
    std::ofstream fout(filename.c_str());

    if(!fout)
    {
        std::cout<<"Cannot open file:"<<filename <<std::endl;  
        return;
    }

    switch( m.type() ) 
    {
        case CV_32S:
            for(int i=0; i<m.rows; i++){
                for(int j=0; j<m.cols; j++){
                    fout<<m.at<int>(i,j)<<"\t";
                }fout<<std::endl;
            }
            break;

        case CV_32F:
            for(int i=0; i<m.rows; i++){
                for(int j=0; j<m.cols; j++){
                    fout<<m.at<float>(i,j)<<"\t";
                }fout<<std::endl;
            }
            break;

        case CV_64F:
            for(int i=0; i<m.rows; i++){
                for(int j=0; j<m.cols; j++){
                    fout<<m.at<double>(i,j)<<"\t";
                }fout<<std::endl;
            }
            break;
        
        default:
            std::cout<<"This datatype has not been covered yet"<<std::endl;
            CV_Assert(false);
            break;
    }
    fout.close();
}