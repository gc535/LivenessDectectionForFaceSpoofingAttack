#include "math.h" 
#include "LBP.hpp"

void LBP::computeLBPImageByRadius(const cv::Mat &srcImage, cv::Mat &LBPImage, Mode mode, const int radius)
{
    CV_Assert(srcImage.total() > 0);
    CV_Assert(srcImage.depth() == CV_8U && srcImage.channels() == 1);
    
    // LBP pattern at each pixel can be a 32-bit integer
    cv::Mat IntSrcImg;
    LBPImage.create(srcImage.size(), CV_32S);
    srcImage.convertTo(IntSrcImg, CV_32S);

    // convert type and pad the image
    cv::Mat PaddedImg;
    copyMakeBorder(IntSrcImg, PaddedImg, radius, radius, radius, radius, cv::BORDER_DEFAULT);
    // compute raw LBP image
    for(int r = 0; r < srcImage.rows; ++r)
    {
        for(int c = 0; c < srcImage.cols; ++c)
        {
            LBPImage.at<int>(r, c) = getPatternByRadius(r+radius, c+radius, radius, PaddedImg);
        }
    }

    if(mode == RIU2)
    {
        for(int r = 0; r < LBPImage.rows; ++r)
        {
            for(int c = 0; c < LBPImage.cols; ++c)
            {
                int RIU2Pattern =  getHopCount( getRIPattern( LBPImage.at<int>(r, c), radius ), radius );
                
                LBPImage.at<int>(r, c) = RIU2Pattern;
                //LBPImage.at<int>(r, c)
            }
        }
    }
}

//srcImage:image
//LBPImage:LBP
void LBP::computeLBPImage(const cv::Mat &srcImage, cv::Mat &LBPImage, Mode mode)
{
    // params check and allocate memeory
    CV_Assert(srcImage.total() > 0);
    CV_Assert(srcImage.depth() == CV_8U && srcImage.channels() == 1);
    LBPImage.create(srcImage.size(), srcImage.type());

    // padding the image
    cv::Mat extendedImage;
    copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, cv::BORDER_DEFAULT);

    // set up LBP calculation
    int heightOfExtendedImage = extendedImage.rows;
    int widthOfExtendedImage = extendedImage.cols;
    int widthOfLBP=LBPImage.cols;
    
    uchar *rowOfExtendedImage= extendedImage.data+widthOfExtendedImage+1;
    uchar *rowOfLBPImage = LBPImage.data;
    for (int y = 1; y <= heightOfExtendedImage - 2; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBP)
    {
        uchar *colOfExtendedImage = rowOfExtendedImage;
        uchar *colOfLBPImage = rowOfLBPImage;
        for (int x = 1; x <= widthOfExtendedImage - 2; ++x,++colOfExtendedImage,++colOfLBPImage)
        {
            /* this section will do a zero counting on the neighbour pixels,
               it will mark the center pixel if it found more than 3 black pixels
               in neighbours, since it might be caused by 0 padding.
               Comment out this section if you dont want this check */
            if (mode != BASIC)  // zero checking is not compatible with 256 Bins LBP
            {
               int zeroCheck = ( colOfExtendedImage[0] ==0 ) +
                            ( colOfExtendedImage[0 - widthOfExtendedImage - 1] ==0 ) +
                            ( colOfExtendedImage[0 - widthOfExtendedImage] ==0) + 
                            ( colOfExtendedImage[0 - widthOfExtendedImage + 1] ==0 ) +
                            ( colOfExtendedImage[0 + 1] ==0 ) + 
                            ( colOfExtendedImage[0 + widthOfExtendedImage + 1] ==0 ) +
                            ( colOfExtendedImage[0 + widthOfExtendedImage] ==0 ) + 
                            ( colOfExtendedImage[0 + widthOfExtendedImage - 1] ==0) +
                            ( colOfExtendedImage[0 - 1]==0 );
             
                if (zeroCheck>=3)  
                {
                    colOfLBPImage[0] = 255;  //it is a problematic for basic LBP 
                    continue; 
                } 
            }
            /* END of zero checking section */

            // calculate LBP value
            int LBPValue = (  ((colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0]) << 7)
                            | ((colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0])     << 6)
                            | ((colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0]) << 5)
                            | ((colOfExtendedImage[0 + 1] >= colOfExtendedImage[0])                        << 4)
                            | ((colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0]) << 3)
                            | ((colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0])     << 2)
                            | ((colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0]) << 1)
                            | ((colOfExtendedImage[0 - 1] >= colOfExtendedImage[0])                            )  );
            // mapping 
            if(mode==BASIC)  // basic LBP 256 bins
            {
                colOfLBPImage[0] = LBPValue;  
            }
            if(mode==RIU1)  // uniform LBP 58 bins, bin 0 not considered in the feature vector
            {
                colOfLBPImage[0] = _uniform[LBPValue];   
            }  
            if(mode==RIU2)  // rotation invariant uniform LBP 58 bins, bin 0 not considered in the feature vector
            {
                colOfLBPImage[0] = getRIU2Pattern(_uniform[_minRI[LBPValue]]); 	
            }  
        }  // col      
    }// row
}


void LBP::computeLBPFeatureVector(const cv::Mat &srcImage, cv::Mat &featureVector, Mode mode)
{
    LBP::computeLBPFeatureVector(srcImage, featureVector, _cellSize, mode);
}

void LBP::computeLBPFeatureVector(const cv::Mat &srcImage, cv::Mat &featureVector, cv::Size cellSize, Mode mode)
{
    // 参数检查，内存分配
    CV_Assert(srcImage.total() > 0);
    CV_Assert(srcImage.depth() == CV_8U && srcImage.channels() == 1);

    cv::Mat LBPImage;
    computeLBPImage(srcImage, LBPImage, mode);

    // 计算cell个数
    int widthOfCell = cellSize.width;
    int heightOfCell = cellSize.height;
    int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
    int numberOfCell_Y = srcImage.rows / heightOfCell;

    // 特征向量的个数
    int numberOfBins = _binNum[mode];  // 256, 58, 9
    int numberOfDimension = numberOfBins * numberOfCell_X*numberOfCell_Y;
    featureVector.create(1, numberOfDimension, CV_32FC1);
    featureVector.setTo(cv::Scalar(0));

    // 计算LBP特征向量
    int stepOfCell=srcImage.cols;
    int pixelCount = cellSize.width*cellSize.height;
    float *dataOfFeatureVector=(float *)featureVector.data;

    // cell的特征向量在最终特征向量中的起始位置
    int index = -numberOfBins;
    for (int y = 0; y <= numberOfCell_Y - 1; ++y)
    {
        for (int x = 0; x <= numberOfCell_X - 1; ++x)
        {
            index+=numberOfBins;

            // 计算每个cell的LBP直方图
            cv::Mat cell = LBPImage(cv::Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
            uchar *rowOfCell=cell.data;
            int sum = 0;
            for(int y_Cell=0;y_Cell<=cell.rows-1;++y_Cell,rowOfCell+=stepOfCell)
            {
                uchar *colOfCell=rowOfCell;
                for(int x_Cell=0;x_Cell<=cell.cols-1;++x_Cell,++colOfCell)
                {

                    /* this section will do a zero counting on the neighbour pixels,
                       it will mark the center pixel if it found more than 3 black pixels
                       in neighbours, since it might be caused by 0 padding.
                       Comment out this section if you dont want this check */
                    if (mode != BASIC && colOfCell[0]==255)   // zero checking is not compatible with 256 Bins mode
                    {
                       continue;
                    }
                    /* END of zero checking section */

                    if(mode!=BASIC) 
                    {
                        if(colOfCell[0]!=0) {
                          // 在直方图中转化为0~numberOfBins-1，所以是colOfCell[0] - 1
                          ++dataOfFeatureVector[index + colOfCell[0]-1];
                          ++sum;
                        }
                    }
                    else
                    {
                        ++dataOfFeatureVector[index + colOfCell[0]];
                    }
                }
            }
            // normalize
            for (int i = 0; i < numberOfBins; ++i)
            {
                if (mode==BASIC)
                {
                    dataOfFeatureVector[index + i] = dataOfFeatureVector[index + i]/pixelCount;
                }
                else
                {
                    if(sum > 0)
                    {
                        dataOfFeatureVector[index + i] = dataOfFeatureVector[index + i]/sum;
                    }
                }
            }
        }
    }

}

int LBP::getPatternByRadius(const int row, const int col, const int radius, cv::Mat& PaddedImg)
{
    /* 
     iterate through the "ring" defined by radius distance
     from interest point: (row, col)
          -------------
          |           |
          |   (r,c)   |
          |           |
          | _________ |
    */
    if(radius > 3 && radius < 0)
    {
        throw std::range_error( "[ERROR]: Only support LBP radius 0<=x<=4." ); 
    }

    int r, c;
    int pattern = 0;
    // top row
    for(c = col-radius, r = row-radius; c <= col+radius; ++c)
    {
        pattern = (pattern << 1) | (PaddedImg.at<int>(r, c) > PaddedImg.at<int>(row, col));
    } 

    // right col
    for(c = col+radius, r = row-radius+1; r <= row+radius; ++r)
    {
        pattern = (pattern << 1) | (PaddedImg.at<int>(r, c) > PaddedImg.at<int>(row, col));
    }

    // bottom row
    for(c = col+radius-1, r = row+radius; c >= col-radius+1; --c)
    {
        pattern = (pattern << 1) | (PaddedImg.at<int>(r, c) > PaddedImg.at<int>(row, col));
    }

    // left col
    for(c = col-radius, r = row+radius; r >= row-radius+1; --r)
    {
        pattern = (pattern << 1) | (PaddedImg.at<int>(r, c) > PaddedImg.at<int>(row, col));
    }

    return pattern;
}


/*
 find the minimum equivalent rotation invariant representation 
*/
int LBP::getRIPattern(int i, const int radius)
{
    int RIPattern = i;

    for(int shift = 0; shift < radius*8-1; ++shift) // eg. 8 digit only need 7 shifts 
    {
        i = ( (i % 2) << (radius*8-1) ) | (i >> 1);
        RIPattern = min(RIPattern, i);
    }
    return RIPattern;
}

/*
 calculate number of transition from 1-0 or 0-1
 in the binary sequence of i.
*/ 
int LBP::getHopCount(int i, const int radius)
{
    // allocate a space for binary representation
    int a[8*radius] = { 0 };
    int k = 7;
    while (i)
    {
        // check bit
        a[k] = i % 2;
        i/=2;
        --k;
    }

    // get hop count
    int count = 0;
    for (int k = 0; k<8*radius; ++k)
    {
        // check if hitting the left most digit
        if (a[k] != a[k + 1 == 8*radius ? 0 : k + 1])
        {
            ++count;
        }
    }
    return count;

}

// 建立等价模式表
// 这里为了便于建立LBP特征图，58种等价模式序号从1开始:1~58,第59类混合模式映射为0
void LBP::buildUniformPatternTable(int *table, const int radius)
{
    memset(table, 0, 256*sizeof(int));
    uchar temp = 1;
    for (int i = 0; i<256; ++i)
    {
        if (getHopCount(i, radius) <= 2)
        {
            table[i] = temp;
            temp++;
        }
    }
}

float cosine_distance(cv::Mat &A, cv::Mat &B)
{
    float sumxx=0, sumyy=0, sumxy=0; 
    
    for(int r=0; r<A.rows; r++) {
        for(int c=0; c<B.cols; c++) {
              float x = A.at<float>(r,c);
              float y = B.at<float>(r,c);
              sumxx += x*x;
              sumyy += y*y;
              sumxy += x*y;
        } 
    }
    return sumxy/sqrt(sumxx*sumyy); 
}


cv::Mat sobel_image(cv::Mat &src) 
{
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  cv::Mat grad,src_gray = src;

  cv::GaussianBlur( src, src_gray, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
 
  //cv::cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );
  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;

  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  cv::Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  cv::Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_x, abs_grad_x );
  cv::convertScaleAbs( grad_y, abs_grad_y );
  cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );


  return grad; 
}



