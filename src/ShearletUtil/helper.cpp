#include <helper.hpp>

namespace Shearlet
{

void gen_x_y_cordinates(int n, cv::Mat& x1n, cv::Mat& y1n, cv::Mat& x2n, cv::Mat& y2n, cv::Mat& D)
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
        if(x_n != x0)
	    {
	        m1.at<float>(i) = 1.0*(y_n - y0)/(x_n - x0);
	    }
        std::vector<double> line_space = linespace(x0, x_n, n);
        cv::Mat xt(line_space);
        for(int j = 0; j < n; ++j)
        {
            if(!flag)
            {
	            y1.at<int>(i, j) = round(m1.at<float>(i) * (xt.at<double>(j) - x0) + y0);  // wrapped by round operation
                x1.at<int>(i, j) = round((xt.at<double>(j)));        // wrapped by round operation
                x2.at<int>(i, j) = y1.at<int>(i, j);
                y2.at<int>(i, j) = x1.at<int>(i, j);
            }
            else
            {
                x1.at<int>(i, j) = (n - 1)/2 + 1;
                y1.at<int>(i, j) = j+1;
                x2.at<int>(i, j) = j+1;
                y2.at<int>(i, j) = (n - 1)/2 + 1;
            }
        }
    }
    //std::cout<< "x1: " << x1 << std::endl;
    //std::cout<< "y1: " << y1 << std::endl;
    //std::cout<< "x2: " << x2 << std::endl;
    //std::cout<< "y2: " << y1 << std::endl;

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

    y2n.at<int>(n-1, 0) = n;
    cv::flip(x1n, x1n, 0);
    cv::flip(y2n, y2n, 0);
    //std::cout<< "y1n: " << y1n << std::endl;
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
            D.at<int>(y1.at<int>(i, j)-1, x1.at<int>(i, j)-1) = D.at<int>(y1.at<int>(i, j)-1, x1.at<int>(i, j)-1) + 1;
        }
    }

    for(int i = 0; i < L; ++i)
    {
        for(int j = 0; j < L; ++j)
        {
            D.at<int>(y2.at<int>(i, j)-1, x2.at<int>(i, j)-1) = D.at<int>(y2.at<int>(i, j)-1, x2.at<int>(i, j)-1) + 1;
        }
    }
    return D;
}

cv::Mat rec_from_pol(cv::Mat& l, int n, cv::Mat& x1, cv::Mat& y1, cv::Mat& x2, cv::Mat& y2, cv::Mat& D)
{
    // https://github.com/lymhust/Fig2_Fig3_Fig4-for-NR-IQA-with-Shearlet-Transform-and-DNN/blob/master/Shearlet-Transform/rec_from_pol.m
    // This funcion re-assembles the radial slice into block.
    //
    // Inputs: l is the radial slice matrix
    //         n is the order of the matrix that is to be re-assembled
    //         x1,y1,x2,y2 are the polar coordinates generated from function
    //                gen_x_y_cord
    //         D is the matrix containing common polar grid points
    // Output: C is the re-assembled block matrix 
    cv::Mat C = cv::Mat::zeros(n, n, CV_32F);
    int option = 0;
    if(option == 1)
    {
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                C.at<float>(y1.at<int>(i, j)-1, x1.at<int>(i, j)-1) = l.at<float>(i, j);
                C.at<float>(y2.at<int>(i, j)-1, x2.at<int>(i, j)-1) = l.at<float>(i+n, j);
            }
        }
        
    }
    else
    {
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                C.at<float>(y1.at<int>(i, j)-1, x1.at<int>(i, j)-1) = C.at<float>(y1.at<int>(i, j)-1, x1.at<int>(i, j)-1) + l.at<float>(i, j);
                C.at<float>(y2.at<int>(i, j)-1, x2.at<int>(i, j)-1) = C.at<float>(y2.at<int>(i, j)-1, x2.at<int>(i, j)-1) + l.at<float>(i+n, j);
            }
        }
    }
    // average common radial grid values
    cv::Mat output;
    cv::divide(C, D, output, 1, C.type()); // scale = 1
    
    //std::cout << "output rows:" << output.rows << ", cols: " << output.cols << std::endl;
    //std::cout << "output:" << output << std::endl;
    return output;
}

cv::Mat windowing(cv::Mat x, const int L, const int c)
{   
    // https://github.com/lymhust/Fig2_Fig3_Fig4-for-NR-IQA-with-Shearlet-Transform-and-DNN/blob/master/Shearlet-Transform/windowing.m
    // This function computes the Meyer window for
    // L decompositions.
    // Inputs:  x - signal to window
    //          L - number of bandpass filters
    //          c - scaling parameter (e.g: c = 1)
    // Outputs: y - y(:,k) is windowed signal x with bandpass k
    int N = x.rows;
    cv::Mat y = cv::Mat::zeros(N, (L+2), CV_32F);  // output
    int T = ceil(N/(float)(L+2));
    
    cv::Mat g = cv::Mat::zeros(1, 2*T, CV_32F);
    for(int i = 0; i < g.cols; ++i)
    {
        float n = (-1) * T/2.0 + (i+1) - 1;   // since i starts from 1 in .m file
        float test = meyer_wind(c*n/T);
        g.at<float>(i) = test;
    }

    for(int i = 1; i < y.cols; ++i)
    {
        int index = 0;
        float k = -1.0*T/2;
        while(k <= 1.5*T-1)
        {
	        int in_sig = floor( fmod( k+(i+1-1)*T, N) ) + 1;  // since j starts from 2 in .m file
            y.at<float>(in_sig-1, i) = g.at<float>(index) * x.at<int>(in_sig-1);
	        k = k+1;
            index++;
        }
    }
    //std::cout << y << std::endl;

    cv::Mat row_sum;
    cv::reduce(y, row_sum, 1, cv::REDUCE_SUM, CV_32F);
    for(int i = 0; i < y.rows; ++i)
    {
        y.at<float>(i, 0) = std::abs(1 - row_sum.at<float>(i));
    }

    return y;
}

float meyer_wind(const float q)
{
    // https://github.com/lymhust/Fig2_Fig3_Fig4-for-NR-IQA-with-Shearlet-Transform-and-DNN/blob/master/Shearlet-Transform/meyer_wind.m
    // This function computes the Meyer window function
    if(-1.0/3+1.0/2 < q && q<1.0/3+1.0/2)
    {
       return 1;
    }
    else if( (1.0/3+1.0/2<=q & q<=2.0/3+1.0/2) | (-2.0/3+1.0/2<=q & q<=1.0/3+1.0/2) )
    {
        float w = 3 * std::abs(q-1.0/2) - 1;
        float z = pow(w, 4) * (35 - 84*w + 70*pow(w, 2) - 20*pow(w, 3));
        return pow( cos(M_PI/2.0*(z)), 2);
    }
    else
    {
        return 0;
    }
}


void shearing_filters_Myer(std::vector<int>& m, std::vector<int>& num, const int L)
{
    // This function computes the shearing filters (wedge shaped) using the Meyer window
    // function.
    //
    // Inputs: m - size of shearing filter matrix desired, m = [m(1),...,m(N)] where
    //             each entry m(j) determines size of shearing filter matrix at scale j. 
    //         num - the parameter determining the number of directions. 
    //               num = [num(1),...,num(N)] where each entry num(j)
    //               determines the number of directions at scale j.  
    //               num(j) ---> 2^(num(j)) + 2 directions.
    //         L - size of the input image ; L by L input image. 
    //
    //
    // Outputs: dshear{j}(:,:,k) - m(j) by m(j) shearing  filter matrix at orientation
    //                             k and scale j.  
    //
    //
    // For example, dshear=shearing_filters_Myer([100 100 180 180],[3 3 4 4],L);
    // produces cell array 'dshear' consisting of 
    //          10 shearing filters (100 by 100) at scale j = 1 (coarse scale)  
    //          10 shearing filters (100 by 100) at scale j = 2 
    //          18 shearing filters (180 by 180) at scale j = 3 
    //          18 shearing filters (180 by 180) at scale j = 4 (fine scale) 

    std::vector<std::vector<cv::Mat> > w_s;
    for(int i = 0; i < num.size(); ++i)
    {
        int n1 = m[i];
        int level = num[i];
        cv::Mat x11, y11, x12, y12, F1;
        gen_x_y_cordinates(n1, x11, y11, x12, y12, F1);
        
        int N = 2*n1;
        int M = pow(2, level) + 2;
        cv::Mat wf = windowing(cv::Mat::ones(N, 1, CV_32S), pow(2, level), 1);  // wf.type is CV_32F


        cv::Mat w = cv::Mat::zeros(n1, n1, CV_32F);
        std::vector<cv::Mat> individual_direction;
        for(int k = 0; k < M; ++k)
        {
            cv::Mat indexed_col = wf.col(k);
            cv::Mat temp = indexed_col * cv::Mat::ones(N/2, 1, CV_32F);
            individual_direction.push_back( rec_from_pol(temp,n1,x11,y11,x12,y12,F1) );
            w = w + individual_direction[k];
        }

        /*
        for(int k = 0; k < M; ++k)
        {
            individual_direction[k] = Mat_sqrt( individual_direction[k].mul(1/w) );
            //w_s{j}(:,:,k)           = sqrt(     1./w.*w_s{j}(:,:,k));
            //w_s{j}(:,:,k)           = real(fftshift(ifft2(ifftshift((w_s{j}(:,:,k))))));
        } 
        */
    }

}


cv::Mat Mat_sqrt(cv::Mat input)
{
    cv::Mat result = cv::Mat::zeros(input.rows, input.cols, CV_32F);
    for(int r = 0; r < input.rows; ++r)
    {
        for(int c = 0; c < input.cols; ++c)
        {
            result.at<float>(r, c) = sqrt( input.at<float>(r, c) );
        }
    }
    return result;
}




}
