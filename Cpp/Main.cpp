#include "Header.h"
#include <filesystem>

int main(int argc, char* argv[]) {
	//std::cout << argc << "\n";
	if (argc > 1) {
		std::cout << "has args\n";
		Chronograph time;
		for (size_t i = 1; i < argc; i++) {
			std::cout << i << ": " << argv[i] << "\n";
			frontier_from_wav(argv[i]);
		}
		time.stop();
	}
	else {
		std::cout << "no args " << argv[0] << "\n";
		Chronograph time;
		std::string path;
		for (auto& p : std::filesystem::recursive_directory_iterator("./")) {
			if (p.path().extension() == ".wav") {
				path = p.path().stem().string() + ".wav";
				std::cout << path << '\n';
				frontier_from_wav(path);
			}
		}
		time.stop();
	}
	return 0;
}

