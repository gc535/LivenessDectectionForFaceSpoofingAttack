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

# binary for data merging
COMBINE = ${PROJECT_HOME}/bin/combine

_COMBINE_SRC = combine.cpp
_COMBINE_OBJ = $(_COMBINE_SRC:.cpp=.o)
COMBINE_OBJ = $(patsubst %, $(ODIR)/%,$(_COMBINE_OBJ))


.PHONY: all lbpdata mergeddata

all: lbpdata mergeddata

lbpdata: $(LBP)

mergeddata: $(COMBINE)

$(LBP): $(LBP_OBJS) $(COMMON_OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 


$(COMBINE): $(COMBINE_OBJ) $(COMMON_OBJS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 


$(ODIR)/%.o: %.cpp
	$(CXX) -c -g -o $@ $< $(CFLAGS)

print-%  : ; @echo $* = $($*)


clean:
	rm -f $(COMBINE) $(LBP) ./obj/*.o

