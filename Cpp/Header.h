#pragma once
#include <iostream>
#include <string>
#include <vector>
//#include <assert.h>
#include <math.h>
//#include <cmath>
#include <chrono> 
#include <fstream>
#include <algorithm>
//#include <complex>
#include <sndfile.hh> // Wav in and out
//#include "mkl_dfti.h" // Intel MKL
//#include <random>
#include <tuple>

const float PI = 3.14159265358979323846;
//const std::complex<float> I = sqrt(-1.0);



struct point {
	float x;
	float y;
};

struct frontier {
	std::vector<int> X;
	std::vector<float> Y;
};

class Chronograph {
private:
	std::chrono::time_point< std::chrono::steady_clock> start, end;
	float duration = 0.0;
public:
	Chronograph() {
		start = std::chrono::high_resolution_clock::now();
	}
	int stop(std::string message = "Time = ") {
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
	//float a;
	std::vector<float> W;

	Wav(std::vector<float> W_, int fps_=44100) {
		fps = fps_;
		W = W_;
		//n = W.size();
		//a = 0.0;
		//for (size_t i = 0; i < n; i++) {
		//	if (std::abs(W[i]) > a) {
		//		a = std::abs(W[i]);
		//	}
		//}
		std::cout << " n: " << W.size() << " fps: " << fps << "\n";
	}
	//int get_size() {
	//	return n;
	//}
	//int get_fps() {
	//	return fps;
	//}
	//int get_amplitude() {
	//	return a;
	//}
	//const std::vector<float>& get_samples() {
	//	return W;
	//}
	//void set_samples(std::vector<float> W_) {
	//	W = W_;
	//}
	//std::vector<float> get_normalized_samples() {
	//	std::vector<float> N(n);
	//	for (size_t i = 0; i < n; i++) {
	//		N[i] = W[i] / a;
	//	}
	//	return N;
	//}
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
		sf_write_float(outfile, &W[0], W.size());
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
	//int fmt = file.format();
	//int pcm = sf_command(NULL, SFC_GET_FORMAT_SUBTYPE,&fmt, sizeof(fmt));
	std::cout << "Successfully opened file at:" << path << "\n";
	//std::cout << "\n  fps:" << fps;
	//std::cout << "\n  pcm:" << pcm;
	//std::cout << "\n  len:" << n << "\n";

	std::vector<float> W(n);
	file.read(&W[0], n);
	return Wav(W, fps);
};

point get_circle(float x0, float y0, float x1, float y1, float r) {
	//float radsq = r * r;
	float q = sqrt(pow(x1 - x0, 2.0) + pow(y1 - y0, 2.0));
	float c = sqrt(r * r - pow(q / 2.0, 2.0));

	float x3 = (x0 + x1) / 2.0;
	float y3 = (y0 + y1) / 2.0;
	float xc, yc;

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

int argmax(const std::vector<float>& V, const int x0, const int x1) {
	float max = std::abs(V[x0]);
	int idx = x0;
	for (size_t i = x0; i < x1; i++) {
		if (std::abs(V[i]) > max) {
			max = std::abs(V[i]);
			idx = i;
		}
	}
	return idx;
}

void get_pulses(const std::vector<float>& W, std::vector<int>& posX,  std::vector<int>& negX) {
	int n = W.size();
	int sign = sgn(W[0]);
	int x = 1;
	int x0 = 0;
	while (sgn(W[x]) == sign) {
		x++;
	}
	x0 = x + 1;
	sign = sgn(W[x0]);
	int xp = 0;
	//std::cout << "opa " << x0 << "\n";
	for (int x1 = x0; x1 < n; x1++) {
		if (sgn(W[x1])!=sign) {
			if (x1 - x0 > 2) {
				xp = argmax(W, x0, x1);
				if (sgn(W[xp]) >= 0) {
					//std::cout << "pos " << xp << "\n";
					posX.push_back(xp);
					//posY.push_back(W[xp]);
				}
				else {
					//std::cout << "neg " << xp << "\n";
					negX.push_back(xp);
					//negY.push_back(W[xp]);
				}
			}
			x0 = x1;
			sign = sgn(W[x1]);
		}
	}
	//std::cout << "opa " << x0 << "\n";
	return;
}

std::vector<int> get_frontier(const std::vector<float>& W, const std::vector<int>& X) {
	int n = X.size();
	float sumY = 0.0;
	float sumY_vec = 0.0;
	float sumX_vec = 0.0;
	for (int i = 0; i < n - 1; i++) {
		sumY += W[X[i]];
		sumX_vec += X[i + 1] - X[i];
		sumY_vec += W[X[i + 1]] - W[X[i]];
	}
	sumY += W[X[n - 1]];
	float scaling = (sumX_vec / 2.0) / sumY;
	float m0 = scaling * sumY_vec / sumX_vec;
	float sumk = 0.0;
	float mm = sqrt(m0 * m0 + 1);
	float x;
	float y;
	std::vector<float> Y(n);
	Y[0] = W[X[0]] * scaling;
	for (int i = 1; i < n; i++) {
		Y[i] = W[X[i]] * scaling;
		x = X[i] - X[i - 1];
		y = Y[i] - Y[i - 1];
		sumk += -(m0 * x - y) / (x * mm * sqrt(x * x + y * y));
	}
	float r = 1.0 / (sumk / (n - 1));
	float rr = r * r;
	//std::cout << "m0: " << m0 << " k: " << sumk << " n: " << n << " r: " << r << " Y0: " << W[X[0]] * scaling << "\n";
	int idx1 = 0;
	int idx2 = 1;
	std::vector<int> frontierX = { X[0] };
	point pc;
	bool empty;
	while (idx2 < n) {
		pc = get_circle(X[idx1], Y[idx1], X[idx2], Y[idx2], r);
		empty = true;
		for (int i = idx2 + 1; i < n; i++) {
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

void write_frontier(const std::vector<float>& W, const std::vector<int>& X, std::string path = "teste.csv") {
	std::ofstream out(path);
	for (int i = 0; i < X.size(); ++i) {
		out << X[i] << "," << W[X[i]] << "\n";
	}
	out.close();
}


//PUBLIC

extern "C" __declspec(dllexport) int frontier_from_wav(std::string path);