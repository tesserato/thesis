#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <chrono> 
#include <fstream>
#include <algorithm>
#include <sndfile.hh> // Wav in and out
//#include "mkl_dfti.h" // Intel MKL
#include <tuple>

const double PI = 3.14159265358979323846;

struct point {
	double x;
	double y;
};

//struct frontier {
//	std::vector<int> X;
//	std::vector<double> Y;
//};

class Chronograph {
private:
	std::chrono::time_point< std::chrono::steady_clock> start, end;
	double duration{ 0.0 };
public:
	Chronograph() {
		start = std::chrono::high_resolution_clock::now();
	}
	double stop(std::string message = "Time = ") {
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		if (duration < 1000) {
			std::cout << message << duration << " milliseconds\n";
			return duration;
		}
		if (duration < 60000) {
			duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
			std::cout << message << duration << " seconds\n";
			return duration;
		}
		duration = std::chrono::duration_cast<std::chrono::minutes>(end - start).count();
		std::cout << message << duration << " minutes\n";
		return duration;
	}
};

class Wav {
public:
	int fps;
	std::vector<double> W;

	Wav(std::vector<double> W_, int fps_=44100) {
		fps = fps_;
		W = W_;
		std::cout << " n: " << W.size() << " fps: " << fps << "\n";
	}
	void write(std::string path = "test.wav") {
		if (W.size() == 0) {
			std::cout << "size = 0";
			return;
		}
		const char* fname = path.c_str();
		SF_INFO sfinfo;
		sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
		sfinfo.channels = 1;
		sfinfo.samplerate = fps;

		SNDFILE* outfile = sf_open(fname, SFM_WRITE, &sfinfo);
		sf_write_double(outfile, &W[0], W.size());
		sf_close(outfile);
		std::cout << "file written. path=" << path << ", fps=" << fps << "\n";
	}
};

Wav read_wav(std::string path) {
	const char* fname = path.c_str();
	SF_INFO sfinfo;
	sfinfo.format = 0;
	SNDFILE* test = sf_open(fname, SFM_READ, &sfinfo);

	if (test == NULL) {
		std::cout << "Couldn't open file at:" << fname;
		std::cout << "\n" << sf_strerror(test);
		sf_close(test);
		exit(1);
	}
	sf_close(test);

	SndfileHandle file = SndfileHandle(fname);

	if (file.channels() != 1) {
		std::cout << "Unexpected number of channels:" << file.channels();
		exit(1);
	}

	int fps = file.samplerate();
	static int n = file.frames();
	std::cout << "Successfully opened file at:" << path << "\n";

	std::vector<double> W(n);
	file.read(&W[0], n);
	return Wav(W, fps);
};

point get_circle(double x0, double y0, double x1, double y1, double r) {
	double q{ sqrt(pow(x1 - x0, 2.0) + pow(y1 - y0, 2.0)) };
	double c{ sqrt(r * r - pow(q / 2.0, 2.0)) };

	double x3{ (x0 + x1) / 2.0 };
	double y3{ (y0 + y1) / 2.0 };
	double xc, yc;

	if (y0 + y1 >= 0) {
		xc = x3 + c * (y0 - y1) / q;
		yc = y3 + c * (x1 - x0) / q;
	}
	else {
		xc = x3 - c * (y0 - y1) / q;
		yc = y3 - c * (x1 - x0) / q;
	}
	return { xc, yc };
}

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

int argmax(const std::vector<double>& V, const int x0, const int x1) {
	double max{ std::abs(V[x0]) };
	int idx = x0;
	for (size_t i = x0; i < x1; i++) {
		if (std::abs(V[i]) > max) {
			max = std::abs(V[i]);
			idx = i;
		}
	}
	return idx;
}

void get_pulses(const std::vector<double>& W, std::vector<size_t>& posX,  std::vector<size_t>& negX) {
	size_t n{ W.size() };
	int sign{ sgn(W[0]) };
	int x{ 1 };
	int x0{ 0 };
	while (sgn(W[x]) == sign) {
		x++;
	}
	x0 = x + 1;
	sign = sgn(W[x0]);
	int xp{ 0 };
	for (size_t x1 = x0; x1 < n; x1++) {
		if (sgn(W[x1])!=sign) {
			if (x1 - x0 > 2) {
				xp = argmax(W, x0, x1);
				if (sgn(W[xp]) >= 0) {
					//std::cout << "pos " << xp << "\n";
					posX.push_back(xp);
				}
				else {
					//std::cout << "neg " << xp << "\n";
					negX.push_back(xp);
				}
			}
			x0 = x1;
			sign = sgn(W[x1]);
		}
	}
	//std::cout << "opa " << x0 << "\n";
	return;
}

std::vector<size_t> get_frontier(const std::vector<double>& W, const std::vector<size_t>& X) {
	size_t n{ X.size() };
	double sumY{ 0.0 };
	double sumY_vec{ W[X[n-1]] - W[X[0]] };
	size_t sumX_vec{ X[n-1] - X[0] };

	//std::cout << "HERE 01!\n";
	for (size_t i = 0; i < n; i++) {
		sumY += W[X[i]];
	}
	double scaling{ (sumX_vec / 2.0) / sumY };
	//double m0{ scaling * sumY_vec / sumX_vec };0
	double sumk{ 0.0 };
	//double mm{ sqrt(m0 * m0 + 1 )};1
	double x;
	double y;

	std::vector<double> Y(n);
	Y[0] = W[X[0]] * scaling;
	//std::cout << "HERE 02!\n";
	for (size_t i = 1; i < n; i++) {
		Y[i] = W[X[i]] * scaling;
		x = X[i] - X[i - 1];
		y = Y[i] - Y[i - 1];
		sumk += y / (x * sqrt(x * x + y * y));
	}
	//std::cout << "HERE 03!\n";
	double r{ 1.0 / (sumk / (n - 1)) };
	double rr{ r * r };
	size_t idx1{ 0 };
	size_t idx2{ 1 };
	std::vector<size_t> frontierX = { X[0] };
	point pc;
	bool empty;
	//std::cout << "HERE 04!\n";
	while (idx2 < n) {
		pc = get_circle(X[idx1], Y[idx1], X[idx2], Y[idx2], r);
		empty = true;
		for (size_t i = idx2 + 1; i < n; i++) {
			if (pow(pc.x - X[i], 2.0) + pow(pc.y - Y[i], 2.0) < rr) {
				empty = false;
				idx2 ++;
				break;
			}
		}
		if (empty) {
			frontierX.push_back(X[idx2]);
			idx1 = idx2;
			idx2 ++;
		}
	}
	return frontierX;
}

void write_frontier(const std::vector<double>& W, const std::vector<size_t>& X, std::string path = "teste.csv") {
	std::ofstream out(path);
	for (size_t i = 0; i < X.size(); ++i) {
		out << X[i] << "," << W[X[i]] << "\n";
	}
	out.close();
}


//PUBLIC

//extern "C" __declspec(dllexport) int frontier_from_wav(std::string path);