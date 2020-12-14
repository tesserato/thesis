#pragma once
#include <stdexcept>
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
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

const double PI = 3.14159265358979323846;

static bool abs_compare(double a, double b) {
	return (std::abs(a) < std::abs(b));
}

struct point {
	double x;
	double y;
};

struct pulse {
	size_t start;
	size_t end;
	pulse(int s, int e) {
		if (s < 0) {
			std::cout << "negative start for Pulse\n";
		}
		if (e < 0) {
			std::cout << "negative end for Pulse\n";
		}
		start = s;
		end = e;
	}
};

struct mode_abdm { // mode & average absolute deviation from mode
	size_t mode;
	double abdm;
	mode_abdm() {
		mode = 0;
		abdm = 0.0;
	}
	mode_abdm(size_t m, double a) {
		mode = m;
		abdm = a;
	}
};

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

class Compressed {
public:
	std::vector<size_t> X_PCs; // start of each pulse
	std::vector<double> Envelope;
	std::vector<double> Waveform;
	Compressed(std::vector<size_t> Xp, std::vector<double> W, const std::vector<double>& S) {
		X_PCs = Xp;
		Waveform = W; // TODO
		for (size_t i = 0; i < Xp.size() - 1; i++) {
			Envelope.push_back(std::abs(*std::max_element(S.begin()+ Xp[i], S.begin()+ Xp[i + 1], abs_compare)));
		}
	}
	Wav reconstruct(int fps = 44100) {
		std::vector<double> Wave;
		for (double & e : Envelope) {
			for (size_t i = 0; i < Waveform.size() - 1; i++) {
				Wave.push_back(e * Waveform[i]);
			}
		}
		return Wav(Wave, fps);
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

int argabsmax(const std::vector<double>& V, const int x0, const int x1) {
	double max{ std::abs(V[x0]) };
	int idx = x0;
	for (size_t i = x0; i <= x1; i++) {
		if (std::abs(V[i]) > max) {
			max = std::abs(V[i]);
			idx = i;
		}
	}
	return idx;
}

int argmax(const std::vector<double>& V, const int x0, const int x1) {
	double max{ std::abs(V[x0]) };
	int idx = x0;
	for (size_t i = x0; i <= x1; i++) {
		if (V[i] > max) {
			max = V[i];
			idx = i;
		}
	}
	return idx;
}

int argmin(const std::vector<double>& V, const int x0, const int x1) {
	double min{ std::abs(V[x0]) };
	int idx = x0;
	for (size_t i = x0; i <= x1; i++) {
		if (V[i] < min) {
			min = V[i];
			idx = i;
		}
	}
	return idx;
}

//int argextremum(const std::vector<double>& V, const int x0, const int x1, const double sign) {
//	//std::cout << sign << "\n";
//	int idx = 0;
//	if (sign >= 0) {
//		double max{ V[0] };
//		for (int i = x0; i < x1; i++) {
//			if (V[i] > max) {
//				max = V[i];
//				idx = i;
//			}
//		}
//	} 
//	else {
//		double min{ V[0] };
//		for (int i = x0; i < x1; i++) {
//			if (V[i] < min) {
//				min = V[i];
//				idx = i;
//			}
//		}
//	}
//	return idx;
//}

std::vector<size_t> find_zeroes(const std::vector<double>& W) {
	int sign{ sgn(W[0]) };
	std::vector<size_t> id_of_zeroes;

	for (size_t i = 0; i < W.size(); i++) {
		if (sgn(W[i]) != sign) {
			id_of_zeroes.push_back(i);
			sign = sgn(W[i]);
		}
	}
	return id_of_zeroes;
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
				xp = argabsmax(W, x0, x1);
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

void refine_frontier(std::vector<pulse>& Pulses, const std::vector<size_t>& Xp, const std::vector<double>& W, int avgL, double stdL, int n_stds = 3) {
	std::vector<pulse> Pulses_to_split;
	std::vector<pulse> Pulses_to_test;
	std::vector<size_t> Xzeroes;
	std::function<int(const std::vector<double>& V, const int x0, const int x1)> arg;
	if (W[Xp[0]] >= 0) {
		arg = argmax;
	}
	else {
		arg = argmin;
	}
	int currsign, x;
	std::cout << "Refining frontier; TH=" << avgL + n_stds * stdL << "\n";
	for (size_t i = 1; i < Xp.size(); i++) {
		if (Xp[i] - Xp[i - 1] >= avgL + n_stds * stdL) {
			Pulses_to_split.push_back(pulse(Xp[i - 1], Xp[i]));
		}
	}
	while (Pulses_to_split.size() > 0) {		
		for (pulse p : Pulses_to_split) {
			Xzeroes.clear();
			currsign = sgn(W[p.start]) ;
			for (size_t i = p.start + 1; i < p.end; i++)	{
				//if (currsign == 0) {
				//	Xzeroes.push_back(i - 1);
				//	currsign = sgn(W[i]);
				//}
				if (currsign != sgn(W[i])){
					Xzeroes.push_back(i);
					currsign = sgn(W[i]);
				}
			}
			if (Xzeroes.size() > 1) {
				x = arg(W, Xzeroes[0], Xzeroes.back());
				Pulses_to_test.push_back(pulse(p.start, x));
				Pulses_to_test.push_back(pulse(x, p.end));
			}
		}
		//std::cout << "Pulses_to_split size:" << Pulses_to_split.size() << "\n";
		Pulses_to_split.clear();


		for (pulse p : Pulses_to_test) {
			if (p.end - p.start >= avgL + n_stds * stdL) {
				Pulses_to_split.push_back(p);
			} else {
				Pulses.push_back(p);
			}
		}
		//std::cout << "Pulses_to_test size:" << Pulses_to_test.size() << "\n";
		Pulses_to_test.clear();
	}
}

mode_abdm get_mode_and_abdm(std::vector<size_t>& T) {
	std::sort(T.begin(), T.end());
	size_t curr_value = T[0];
	size_t curr_count = 0;
	size_t max_count = 0;
	size_t mode = 0;
	for (auto t: T) {
		if (t == curr_value){
			curr_count++;
		} 
		else {
			if (curr_count > max_count) {
				max_count = curr_count;
				mode = curr_value;
			}
			curr_value = t;
			curr_count = 1;
		}
	}
	if (curr_count > max_count) {
		max_count = curr_count;
		mode = curr_value;
	}
	int abdm{ 0 };
	for (auto t : T) {
		abdm += std::abs(int(t) - int(mode));
	}
	
	return mode_abdm(mode, double(abdm) / double(T.size()));
}

void refine_frontier_iter(std::vector<size_t>& Xp, const std::vector<double>& W) {
	std::vector<size_t> T(Xp.size() - 1);
	for (size_t i = 0; i < Xp.size() - 1; i++)	{
		T[i] = Xp[i + 1] - Xp[i];
	}

	mode_abdm modeabdm = get_mode_and_abdm(T);
	size_t mde{ modeabdm.mode };
	double std{ modeabdm.abdm };

	std::vector<pulse> Pulses;
	refine_frontier(Pulses, Xp, W, mde, std);

	size_t psize{ 0 };
	double std_c{ 0.0 };
	while (Pulses.size() > psize) {
		std::cout << "Xp size before:" << Xp.size() << "\n";
		psize = Pulses.size();
		for (pulse p:Pulses)	{
			Xp.push_back(p.start);
			Xp.push_back(p.end);
		}
		std::vector<size_t>::iterator it=std::unique(Xp.begin(), Xp.end());
		Xp.resize(std::distance(Xp.begin(), it));
		std::cout << "Xp size after:" << Xp.size() << "\n";
		T.resize(Xp.size() - 1);
		for (size_t i = 0; i < Xp.size() - 1; i++) {
			T[i] = Xp[i + 1] - Xp[i];
		}

		modeabdm = get_mode_and_abdm(T);
		mde = modeabdm.mode;
		std_c = modeabdm.abdm;

		if (std_c < std) {
			std = std_c;
			refine_frontier(Pulses, Xp, W, mde, std);
		}
		else {
			std::cout << "std0=" << std << " , std1=" << std_c << "\n";
			break;
		}
	}
}

//void write_frontier(const std::vector<double>& W, const std::vector<size_t>& X, std::string path = "teste.csv") {
//	std::ofstream out(path);
//	for (size_t i = 0; i < X.size(); ++i) {
//		out << X[i] << "," << W[X[i]] << "\n";
//	}
//	out.close();
//}

std::vector<size_t> get_Xpcs(const std::vector<size_t>& Xpos, const std::vector<size_t>& Xneg) {
	size_t min_id{ std::min(Xpos.size(),Xneg.size()) };
	std::vector<size_t> Xpcs(min_id);
	for (size_t i = 0; i < min_id; i++)	{
		Xpcs[i] = std::round((float(Xpos[i]) + float(Xneg[i])) / 2);
	}
	std::vector<size_t>::iterator it = std::unique(Xpcs.begin(), Xpcs.end());
	Xpcs.resize(std::distance(Xpcs.begin(), it));
	std::sort(Xpcs.begin(), Xpcs.end());
	return Xpcs;
}

mode_abdm average_pc_waveform(std::vector<double>& pcw, const std::vector<size_t>& Xp, const std::vector<double>& W) {
	std::vector<size_t> T(Xp.size() - 1);
	for (size_t i = 0; i < Xp.size() - 1; ++i) {
		T[i] = Xp[i + 1] - Xp[i];
	}

	mode_abdm modeabdm = get_mode_and_abdm(T);
	size_t mode{ modeabdm.mode };

	int x0{ 0 };
	int x1{ 0 };
	double step{ 1.0 / double(mode) };
	pcw.resize(mode + 1);
	std::fill(pcw.begin(), pcw.end(), 0.0);

	for (size_t i = 0; i < Xp.size() - 1; i++) {
		x0 = Xp[i];
		x1 = Xp[i + 1];
		if (x1 - x0 > 5) {
			boost::math::interpolators::cardinal_cubic_b_spline<double> spline(W.begin() + x0, W.begin() + x1, 0, 1.0 / float(x1 - x0));
			for (size_t i = 0; i <= mode; i++) {
				pcw[i] += spline(i * step);
			}			
		}
	}
	double amp{std::abs(*std::max_element(pcw.begin(), pcw.end(), abs_compare))};
	for (size_t i = 0; i <= mode; i++) {
		pcw[i] = pcw[i] / amp;
	}
	return modeabdm;
}

double average_pc_metric(const std::vector<double>& pcw, const std::vector<size_t>& Xp, const std::vector<double>& W) {
	size_t mode{ pcw.size() };

	int x0{ 0 };
	int x1{ 0 };
	size_t ac{ 0 };
	double step{ 1.0 / double(mode) };
	double avdv{ 0.0 };
	for (size_t i = 0; i < Xp.size() - 1; i++) {
		x0 = Xp[i];
		x1 = Xp[i + 1];
		if (x1 - x0 > 5) {
			boost::math::interpolators::cardinal_cubic_b_spline<double> spline(W.begin() + x0, W.begin() + x1, 0, 1.0 / double(x1 - x0));
			for (size_t i = 0; i <= mode; i++) {
				avdv += std::abs(spline(i * step) - W[x0 + i]);
				ac++;
			}
		}
	}
	avdv += avdv / double(ac);
	return avdv;
}

std::vector<size_t> refine_Xpcs(const std::vector<double>& W, const std::vector<double>& avgpc, size_t min_size, size_t max_size) {
	if (avgpc.size() <= 5) {
		std::cout << "Average Pseudo Cycle waveform size <= 5";
		throw std::invalid_argument("Average Pseudo Cycle waveform size <= 5");
	}
	boost::math::interpolators::cardinal_cubic_b_spline<double> spline(avgpc.begin(), avgpc.end(), 0, 1.0 / avgpc.size());
	std::vector<double> Wpadded(W.size() + 2 * min_size, 0.0);
	for (size_t i = min_size; i < W.size() + min_size; i++) {
		Wpadded[i] = W[i - min_size];
	}
	size_t best_size{ 0 };
	size_t best_x0{ 0 };
	double best_val{ 0.0 };
	double curr_val{ 0.0 };
	double step{ 0.0 };
	double amp{0.0};
	std::vector<size_t> Xpc;
	std::vector<std::vector<double>> interps(max_size - min_size+1, std::vector<double>(max_size + 1));
	for (size_t size = min_size; size <= max_size; size++) {
		step = 1.0 / float(size);
		for (size_t x0 = 1; x0 < min_size; x0++) {
			curr_val = 0.0;
			//amp = std::abs(*std::max_element(Wpadded.begin() + x0, Wpadded.begin() + x0 + size, abs_compare));
			for (size_t i = 0; i <= size; i++) {
				interps[size - min_size][i] = spline(i * step);
				curr_val += Wpadded[x0 + i] * interps[size - min_size][i];// / amp;

			}
			if (curr_val > best_val) {
				best_val = curr_val;
				best_x0 = x0;
				best_size = size;
			}
		}
	}
	int x0{ int(best_x0) - int(min_size) };
	//std::cout << "best c:" << x0 << "\n";
	if (x0 > 0) {
		Xpc.push_back(x0);
	}
	size_t curr_x0{ best_x0 + best_size };

	x0 = int(curr_x0) - int(min_size);

	if (x0 > 0) {
		Xpc.push_back(x0);
	}
	
	while (curr_x0 + max_size - min_size < W.size()) {
		best_val = 0.0;
		for (size_t size = min_size; size <= max_size; size++) {
			step = 1.0 / float(size);
			curr_val = 0.0;
			//amp = std::abs(*std::max_element(Wpadded.begin() + curr_x0, Wpadded.begin() + curr_x0 + size, abs_compare));
			for (size_t i = 0; i <= size; i++) {
				curr_val += Wpadded[curr_x0 + i] * interps[size - min_size][i];// / amp;
			}
			if (curr_val > best_val) {
				best_val = curr_val;
				best_size = size;
			}
		}
		curr_x0 += best_size;
		//std::cout << "c:" << curr_x0 << "\n";
		Xpc.push_back(curr_x0 - min_size);
	}
	return Xpc;
}