PROJECT_HOME = .
OPENCV_PATH=/usr/local/lib
CUXX = g++

BIN = ${PROJECT_HOME}/bin

SRCS = ./src/LBP.cpp  
OBJDIR = ${PROJECT_HOME}/obj
OBJS = $(SRCS:.cpp=.o)

CFLAGS = -O2 -std=c++11 -I${PROJECT_HOME}/src/ 
LIBFLAGS = -L${PROJECT_HOME}/src -L${OPENCV_PATH} -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_ml

all: $(BIN)

$(OBJS):

%.o: %.c
	$(CUXX) -c -g $(CFLAGS) $< -o $(OBJDIR)/$@ $<

$(BIN): $(OBJS)
	$(CUXX) $^ $(LIBFLAGS) -o $@ 

clean:
	rm -f $(BIN) ./obj/*.o

