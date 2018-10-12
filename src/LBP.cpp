#include "math.h" 
#include "LBP.hpp"


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
                dataOfFeatureVector[index + i] = (mode==BASIC) ? 
                    (dataOfFeatureVector[index + i]/pixelCount) : (dataOfFeatureVector[index + i]/sum);
            }
        }
    }

}


//获取i中0,1的跳变次数
int LBP::getHopCount(int i)
{
    // 转换为二进制
    int a[8] = { 0 };
    int k = 7;
    while (i)
    {
        // 除2取余
        a[k] = i % 2;
        i/=2;
        --k;
    }

    // 计算跳变次数
    int count = 0;
    for (int k = 0; k<8; ++k)
    {
        // 注意，是循环二进制,所以需要判断是否为8
        if (a[k] != a[k + 1 == 8 ? 0 : k + 1])
        {
            ++count;
        }
    }
    return count;

}

// 建立等价模式表
// 这里为了便于建立LBP特征图，58种等价模式序号从1开始:1~58,第59类混合模式映射为0
void LBP::buildUniformPatternTable(int *table)
{
    memset(table, 0, 256*sizeof(int));
    uchar temp = 1;
    for (int i = 0; i<256; ++i)
    {
        if (getHopCount(i) <= 2)
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



