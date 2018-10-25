#ifndef COMBINE_HPP
#define COMBINE_HPP

#include <stdlib.h>
#include <string>

void ParseArgument(const int& argc, const char* const* argv, std::string& model_1, std::string& checkpoint_1, 
				  std::string& model_2, std::string& checkpoint_2, int& resize, int& cell_size,
				  std::string& data_path);
void printHelp();


#endif //COMBINE_HPP