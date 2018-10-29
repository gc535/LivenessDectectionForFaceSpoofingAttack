#ifndef OFM_HPP
#define OFM_HPP

#include <stdlib.h>
#include <string>

void ParseArgument(const int& argc, const char* const* argv,  
				   int& resize, int& cell_size,
				   std::string& data_path);
void printHelp();


#endif //OFM_HPP