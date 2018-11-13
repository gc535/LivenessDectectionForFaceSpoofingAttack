PROJECT_HOME=..
CXX = g++
IDIR=.
CAFFE_IDIR=/home/ubuntu/anaconda3/envs/caffe-cpu/include
CFLAGS = -O0 -std=c++11 -I$(IDIR) 

#OPENCV_PATH=/usr/local/lib
OPENCV_PATH=/usr/local/lib

LIBFLAGS = -L${OPENCV_PATH} -lopencv_hdf -lopencv_ml -lopencv_dnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core 
ODIR=obj


_COMMON_SRCS = Data.cpp LBP.cpp Util.cpp SingleChannelDOG.cpp
_COMMON_OBJS = $(_COMMON_SRCS:.cpp=.o)
COMMON_OBJS = $(patsubst %, $(ODIR)/%,$(_COMMON_OBJS))


# binary for lbp extraction
LBP = ${PROJECT_HOME}/bin/lbp

_LBP_SRCS := lbpExtraction.cpp
_LBP_OBJS := $(_LBP_SRCS:.cpp=.o)
LBP_OBJS := $(patsubst %, $(ODIR)/%,$(_LBP_OBJS))

# binary for ofm data preparation
OFM = ${PROJECT_HOME}/bin/ofm

_OFM_SRCS := OFM.cpp
_OFM_OBJS := $(_OFM_SRCS:.cpp=.o)
OFM_OBJS := $(patsubst %, $(ODIR)/%,$(_OFM_OBJS))

# binary for fft_dog data preparation
FFT_DOG = ${PROJECT_HOME}/bin/fft_dog

_FFT_DOG_SRCS := FFT_DOG.cpp
_FFT_DOG_OBJS := $(_FFT_DOG_SRCS:.cpp=.o)
FFT_DOG_OBJS := $(patsubst %, $(ODIR)/%,$(_FFT_DOG_OBJS))

# binary for data merging
INTEGRATION = ${PROJECT_HOME}/bin/integration

_INTEGRATION_SRC = integration.cpp
_INTEGRATION_OBJ = $(_INTEGRATION_SRC:.cpp=.o)
INTEGRATION_OBJ = $(patsubst %, $(ODIR)/%,$(_INTEGRATION_OBJ))


.PHONY: all lbpdata ofmdata fft_dog integration

all: lbpdata ofmdata fft_dog integration

lbpdata: $(LBP)
ofmdata: $(OFM)
fft_dog: $(FFT_DOG)
integration: $(INTEGRATION)

$(LBP): $(LBP_OBJS) $(COMMON_OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 

$(OFM): $(OFM_OBJS) $(COMMON_OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS)

$(FFT_DOG): $(FFT_DOG_OBJS) $(COMMON_OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS)

$(INTEGRATION): $(INTEGRATION_OBJ) $(COMMON_OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 


$(ODIR)/%.o: %.cpp
	$(CXX) -c -g -o $@ $< $(CFLAGS)

print-%  : ; @echo $* = $($*)


clean:
	rm -f $(INTEGRATION) $(FFT_DOG) $(LBP) $(OFM) ./obj/*.o

