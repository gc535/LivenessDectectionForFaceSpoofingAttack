#ifndef MAIN_HPP
#define MAIN_HPP



enum Action
{
	TRAIN,
	TEST
};


void printHelp() {
	std::cout << "\nUsage: ./main [options]" << std::endl;
	std::cout << "\nOptions:" << std::endl;
	std::cout << "\t-r <int> - Target resize size (default=96)" << std::endl;
	std::cout << "\t-c <int> - Desired cell size for LBP extraction (default=16)" << std::endl;
	std::cout << "\t-d <string> - Specificy a path to the root of data directory: (default=none)" << std::endl;
	std::cout << "\t \t \tData root should contain a two sub folder: 'train' and 'test" << std::endl;
}

#endif //MAIN_HPP