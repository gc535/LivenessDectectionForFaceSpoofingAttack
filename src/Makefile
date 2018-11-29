CXX = g++

PROJECT_HOME = ..
ROOT_IDIR = .
SHEARLET_IDIR = ./ShearletUtil
CAFFE_IDIR=/home/ubuntu/anaconda3/envs/caffe-cpu/include

IDIR =$(ROOT_IDIR) $(SHEARLET_IDIR)


CFLAGS = -O0 -std=c++11 $(foreach d, $(IDIR), -I$d)

#OPENCV_PATH=/usr/local/lib
OPENCV_PATH=/usr/local/lib

LIBFLAGS = -L${OPENCV_PATH} -lopencv_hdf -lopencv_ml -lopencv_dnn -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core 
ODIR=obj


_COMMON_SRCS = Data.cpp LBP.cpp Util.cpp SingleChannelDOG.cpp ELBP.cpp lbp_hist.cpp
_COMMON_OBJS = $(_COMMON_SRCS:.cpp=.o)
COMMON_OBJS = $(patsubst %, $(ODIR)/%,$(_COMMON_OBJS))


_SHEARLET_ADDON := helper.cpp
_SHEARLET_SRCS := $(patsubst %, $(SHEARLET_IDIR)/%,$(_SHEARLET_ADDON))
_SHEARLET_OBJS := $(_SHEARLET_ADDON:.cpp=.o)
SHEARLET_OBJS := $(patsubst %, $(ODIR)/%,$(_SHEARLET_OBJS))

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


# raw lbp histgram
RAW_LBP = ${PROJECT_HOME}/bin/raw_lbp

_RAW_LBP_SRCS := RawLBP.cpp
_RAW_LBP_OBJS := $(_RAW_LBP_SRCS:.cpp=.o)
RAW_LBP_OBJS := $(patsubst %, $(ODIR)/%,$(_RAW_LBP_OBJS))


# combine dog lbp and raw lbp
COMBINED_LBP = ${PROJECT_HOME}/bin/combined_lbp

_COMBINED_LBP_SRCS := combinedLBP.cpp
_COMBINED_LBP_OBJS := $(_COMBINED_LBP_SRCS:.cpp=.o)
COMBINED_LBP_OBJS := $(patsubst %, $(ODIR)/%,$(_COMBINED_LBP_OBJS))



# binary for data merging
INTEGRATION = ${PROJECT_HOME}/bin/integration

_INTEGRATION_SRC = integration.cpp
_INTEGRATION_OBJ = $(_INTEGRATION_SRC:.cpp=.o)
INTEGRATION_OBJ = $(patsubst %, $(ODIR)/%,$(_INTEGRATION_OBJ))


.PHONY: all lbpdata ofmdata fft_dog raw_lbp combined_lbp integration

all: lbpdata ofmdata fft_dog raw_lbp combined_lbp integration

lbpdata: $(LBP)
ofmdata: $(OFM)
fft_dog: $(FFT_DOG)
raw_lbp: $(RAW_LBP)
combined_lbp: $(COMBINED_LBP)
integration: $(INTEGRATION)

$(LBP): $(LBP_OBJS) $(COMMON_OBJS) $(SHEARLET_OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 

$(OFM): $(OFM_OBJS) $(COMMON_OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS)

$(FFT_DOG): $(FFT_DOG_OBJS) $(COMMON_OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS)

$(RAW_LBP): $(RAW_LBP_OBJS) $(COMMON_OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS)

$(COMBINED_LBP): $(COMBINED_LBP_OBJS) $(COMMON_OBJS) 
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS)

$(INTEGRATION): $(INTEGRATION_OBJ) $(COMMON_OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 


$(ODIR)/%.o: %.cpp
	$(CXX) -c -g -o $@ $< $(CFLAGS)

$(ODIR)/%.o: $(_SHEARLET_SRCS)
	$(CXX) -c -g -o $@ $< $(CFLAGS)

print-%  : ; @echo $* = $($*)


clean:
	rm -f $(INTEGRATION) $(COMBINED_LBP) $(RAW_LBP) $(FFT_DOG) $(LBP) $(OFM) ./obj/*.o

