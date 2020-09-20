// dllmain.cpp : Defines the entry point for the DLL application.

#include "pch.h"
#include "Header.h"
//#include <windows.h>
//#define DLLEXPORT extern "C" __declspec(dllexport)

//char  const char* cpath
extern "C" __declspec(dllexport) int frontier_from_wav(std::string path) {
	//std::cout << "running\n";

	//std::string path(cpath);

	Wav wav = read_wav(path);

	std::string delimiter = "/";

	size_t pos = 0;
	std::string token;
	while ((pos = path.find(delimiter)) != std::string::npos) {
		token = path.substr(0, pos);
		//std::cout << token << std::endl;
		path.erase(0, pos + delimiter.length());
	}
	delimiter = ".wav";
	pos = path.find(delimiter);
	std::string name = path.substr(0, pos);
	//std::cout << name << std::endl;

	std::vector<int> posX;
	//std::vector<float> posY;
	std::vector<int> negX;
	//std::vector<float> negY;

	get_pulses(wav.W, posX, negX);

	std::vector<int> local_posF = get_frontier(wav.W, posX);
	std::vector<int> local_negF = get_frontier(wav.W, negX);

	write_frontier(wav.W, local_posF, name + "_pos_frontier.csv");
	write_frontier(wav.W, local_negF, name + "_neg_frontier.csv");

	return 0;
}


std::vector<int> posF;
std::vector<int> negF;
extern "C" __declspec(dllexport) int compute_raw_envelope(float* cW, unsigned int n) {
	//std::cout << "C : n in = " << n << "\n";
	const std::vector<float> W(cW, cW + n);

	std::vector<int> posX, negX;
	//std::vector<float> posY, negY;

	get_pulses(W, posX, negX);

	posF = get_frontier(W, posX);
	negF = get_frontier(W, negX);

	write_frontier(W, posF, "0_pos_frontier.csv");
	write_frontier(W, negF, "0_neg_frontier.csv");

	return 0;
}

int pos_n, neg_n;

extern "C" __declspec(dllexport) int get_pos_size() {
	pos_n = posF.size();
	//std::cout << "C : pos n = " << pos_n << "\n";
	return pos_n;
}

extern "C" __declspec(dllexport) int* get_pos_X() {
	int* ptr = &posF[0];
	return ptr;
}

//extern "C" __declspec(dllexport) float* get_pos_Y() {
//	float* ptr = &posF.Y[0];
//	return ptr;
//}


extern "C" __declspec(dllexport) int get_neg_size() {
	neg_n = negF.size();
	//std::cout << "C : neg n  = " << neg_n << "\n";
	return neg_n;
}

extern "C" __declspec(dllexport) int* get_neg_X() {
	int* ptr = &negF[0];
	return ptr;
}

//extern "C" __declspec(dllexport) float* get_neg_Y() {
//	float* ptr = &negF.Y[0];
//	return ptr;
//}