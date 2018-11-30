#include <helper.hpp>

void gen_x_y_cordinates(cv::Mat& x1n, cv::Mat& y1n, cv::Mat& x2n, cv::Mat& y2n, cv::Mat& D, int n)
{
    /* 
      This function is a c++ implementation of:
      https://github.com/lymhust/Fig2_Fig3_Fig4-for-NR-IQA-with-Shearlet-Transform-and-DNN/blob/master/Shearlet-Transform/gen_x_y_cordinates.m
    */
    n += 1;
    cv::Mat x1 = cv::Mat::zeros(n, n, CV_32S);
    cv::Mat y1 = cv::Mat::zeros(n, n, CV_32S);
    cv::Mat x2 = cv::Mat::zeros(n, n, CV_32S);
    cv::Mat y2 = cv::Mat::zeros(n, n, CV_32S);
    cv::Mat m1 = cv::Mat::zeros(1, n, CV_32F);

    for(int i = 0; i < n; ++i)
    {
        int y0 = 1, x0 = i+1, x_n = n-i, y_n = n;
        int flag = x_n == x0 ? 1 : 0;
        m1.at<float>(i) = (y_n - y0)/(x_n = x0);
        cv::Mat xt(linespace(x0, x_n, n));
        for(int j = 0; j < n; ++j)
        {
            if(!flag)
            {
                y1.at<int>(i, j) = static_cast<int>(m1.at<float>(i) * (xt.at<double>(j) - x0) + y0);  // wrapped by round operation
                x1.at<int>(i, j) = static_cast<int>((xt.at<double>(j)));        // wrapped by round operation
                x2.at<int>(i, j) = y1.at<int>(i, j);
                y2.at<int>(i, j) = x1.at<int>(i, j);
            }
            else
            {
                x1.at<int>(i, j) = (n - 1)/2 + 1;
                y1.at<int>(i, j) = j;
                x2.at<int>(i, j) = j;
                y2.at<int>(i, j) = (n - 1)/2 + 1;
            }
        }
    }

    n -= 1;
    x1n = cv::Mat::zeros(n, n, CV_32S);
    y1n = cv::Mat::zeros(n, n, CV_32S);
    x2n = cv::Mat::zeros(n, n, CV_32S);
    y2n = cv::Mat::zeros(n, n, CV_32S);
    
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            x1n.at<int>(i, j) = x1.at<int>(i, j);
            y1n.at<int>(i, j) = y1.at<int>(i, j);
            x2n.at<int>(i, j) = x2.at<int>(i+1, j);
            y2n.at<int>(i, j) = y2.at<int>(i+1, j);
        }
    }

    y2n.at<int>(n, 1) = n;
    cv::flip(x1n, x1n, 0);
    cv::flip(y2n, y2n, 0);

    D = avg_pol(n, x1n, y1n, x2n, y2n);
}


std::vector<double> linespace(double start, double end, int num_sample)
{
    std::vector<double> vector;
    double delta = (end - start) / (num_sample - 1); 
    for(int cnt = 0; cnt < num_sample; ++cnt)
    {
        vector.push_back(start + cnt*delta);
    }
    return vector;
}

cv::Mat avg_pol(int L, const cv::Mat& x1, const cv::Mat& y1, const cv::Mat& x2, const cv::Mat& y2)
{   
    // This function generates the matrix that contains the number
    // of times the polar grid points go through the rectangular grid 
    // point i,j
    //
    // Input: L is the order of the block matrix
    //
    // Output: D is the common grid point values
    cv::Mat D = cv::Mat::zeros(L, L, CV_32S);
    for(int i = 0; i < L; ++i)
    {
        for(int j = 0; j < L; ++j)
        {
            D.at<int>(y1.at<int>(i, j), x1.at<int>(i, j)) = D.at<int>(y1.at<int>(i, j), x1.at<int>(i, j)) + 1;
        }
    }

    for(int i = 0; i < L; ++i)
    {
        for(int j = 0; j < L; ++j)
        {
            D.at<int>(y2.at<int>(i, j), x2.at<int>(i, j)) = D.at<int>(y2.at<int>(i, j), x2.at<int>(i, j)) + 1;
        }
    }
    return D;
}

cv::Mat rec_from_pol(cv::Mat& l, int n, cv::Mat& x1, cv::Mat& y1, cv::Mat& x2, cv::Mat& y2, cv::Mat& D)
{
    cv::Mat C = cv::Mat::zeros(n, n, CV_32S);
    int option = 0;
    if(option == 1)
    {
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                C.at<int>(y1.at<int>(i, j), x1.at<int>(i, j)) = l.at<int>(i, j);
                C.at<int>(y2.at<int>(i, j), x2.at<int>(i, j)) = l.at<int>(i+n, j);
            }
        }
        
    }
    else
    {
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                C.at<int>(y1.at<int>(i, j), x1.at<int>(i, j)) = C.at<int>(y1.at<int>(i, j), x1.at<int>(i, j)) + l.at<int>(i, j);
                C.at<int>(y2.at<int>(i, j), x2.at<int>(i, j)) = C.at<int>(y2.at<int>(i, j), x2.at<int>(i, j)) + l.at<int>(i+n, j);
            }
        }
    }
    // average common radial grid values 
    C = C / D;
    return C;
}