// dllmain.cpp : Defines the entry point for the DLL application.

#include "pch.h"
#include "Header.h"

//extern "C" __declspec(dllexport) int frontier_from_wav(std::string path) {
//	//! Takes a path to a mono wav file and writes the envelope as a .csv file
//	/*! This function is to be used inside C++ code */
//
//	Wav wav{ read_wav(path) };
//
//	std::string delimiter{ "/" };
//
//	size_t pos{ 0 };
//	std::string token;
//	while ((pos = path.find(delimiter)) != std::string::npos) {
//		token = path.substr(0, pos);
//		//std::cout << token << std::endl;
//		path.erase(0, pos + delimiter.length());
//	}
//	delimiter = ".wav";
//	pos = path.find(delimiter);
//	std::string name = path.substr(0, pos);
//	//std::cout << name << std::endl;
//
//	std::vector<size_t> posX;
//	//std::vector<float> posY;
//	std::vector<size_t> negX;
//	//std::vector<float> negY;
//
//	get_pulses(wav.W, posX, negX);
//
//	std::vector<size_t> local_posF = get_frontier(wav.W, posX);
//	std::vector<size_t> local_negF = get_frontier(wav.W, negX);
//
//	std::vector<size_t> local_F(local_posF.size() + local_negF.size());
//	std::merge(local_posF.begin(), local_posF.end(), local_negF.begin(), local_negF.end(), local_F.begin());
//
//	//write_frontier(wav.W, local_posF, name + "_pos_frontier.csv");
//	//write_frontier(wav.W, local_negF, name + "_neg_frontier.csv");
//
//	std::ofstream out(name + "_env.csv");
//	out << "idx, W\n";
//	size_t idx;
//	for (size_t i = 0; i < local_F.size(); ++i) {
//		idx = local_F[i];
//		out << idx << "," << wav.W[idx] << "\n";
//	}
//	out.close();
//
//	return 0;
//}

// Python Interface

extern "C" __declspec(dllexport) int test(int p) {
	return p;
}

std::vector<size_t> posF;
std::vector<size_t> negF;
std::vector<size_t> F;
int pos_n, neg_n;

extern "C" __declspec(dllexport) int compute_raw_envelope(double* cW, unsigned int n) {
	//std::cout << "C : n in = " << n << "\n";
	const std::vector<double> W(cW, cW + n);

	std::vector<size_t> posX, negX;
	//std::vector<float> posY, negY;

	get_pulses(W, posX, negX);

	posF = get_frontier(W, posX);
	negF = get_frontier(W, negX);

	//write_frontier(W, posF, "0_pos_frontier.csv");
	//write_frontier(W, negF, "0_neg_frontier.csv");

	F.reserve(posF.size() + negF.size());
	std::merge(posF.begin(), posF.end(), negF.begin(), negF.end(), F.begin());

	//int* ptr = &F[0];

	return 0;
}

extern "C" __declspec(dllexport) int get_pos_size() {
	pos_n = posF.size();
	//std::cout << "C : pos n = " << pos_n << "\n";
	return pos_n;
}

extern "C" __declspec(dllexport) int get_neg_size() {
	neg_n = negF.size();
	//std::cout << "C : neg n  = " << neg_n << "\n";
	return neg_n;
}

extern "C" __declspec(dllexport) size_t * get_pos_X() {
	size_t* ptr = &posF[0];
	return ptr;
}

extern "C" __declspec(dllexport) size_t * get_neg_X() {
	size_t* ptr = &negF[0];
	return ptr;
}

extern "C" __declspec(dllexport) size_t * get_X() {
	size_t* ptr = &F[0];
	return ptr;
}