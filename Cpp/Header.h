#pragma once
#include <iostream>
#include <string>
#include <vector>
//#include <assert.h>
#include <math.h>
#include <cmath>
#include <chrono> 
#include <fstream>
#include <algorithm>
#include <complex>
#include <sndfile.hh> // Wav in and out
#include "mkl_dfti.h" // Intel MKL
#include <random>
#include <tuple>

const float PI = 3.14159265358979323846;
const std::complex<float> I = sqrt(-1.0);

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
private:
	int fps, n;
	float a;
	std::vector<float> W;
public:
	Wav(std::vector<float> W_, int fps_=44100) {
		fps = fps_;
		W = W_;
		n = W.size();
		a = 0.0;
		for (size_t i = 0; i < n; i++) {
			if (std::abs(W[i]) > a) {
				a = std::abs(W[i]);
			}
		}
		std::cout << "Amplitude: " << a << " n: " << n << " fps: " << fps << "\n\n";
	}
	int get_size() {
		return n;
	}
	int get_fps() {
		return fps;
	}
	int get_amplitude() {
		return a;
	}
	std::vector<float> get_samples() {
		return W;
	}
	void set_samples(std::vector<float> W_) {
		W = W_;
	}
	std::vector<float> get_normalized_samples() {
		std::vector<float> N(n);
		for (size_t i = 0; i < n; i++) {
			N[i] = W[i] / a;
		}
		return N;
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
	float q = sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2));
	float x3 = (x0 + x1) / 2;
	float y3 = (y0 + y1) / 2;
	float xc, yc;
	float c = sqrt(r * r - pow(q / 2, 2));
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

void get_pulses(const std::vector<float>& W, std::vector<int>& posX, std::vector<float>& posY, std::vector<int>& negX, std::vector<float>& negY) {
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
				if (sgn(W[xp]) > 0) {
					//std::cout << "pos " << xp << "\n";
					posX.push_back(xp);
					posY.push_back(W[xp]);
				}
				else {
					//std::cout << "neg " << xp << "\n";
					negX.push_back(xp);
					negY.push_back(W[xp]);
				}
			}
			x0 = x1;
			sign = sgn(W[x1]);
		}
	}
	return;
}

frontier get_frontier(const std::vector<int>& X, std::vector<float>& Y) {
	int n = Y.size();
	float sumY = 0.0;
	float sumY_vec = 0.0;
	float sumX_vec = 0.0;
	for (size_t i = 0; i < n - 1; i++) {
		sumY += Y[i];
		sumX_vec += X[i + 1] - X[i];
		sumY_vec += Y[i + 1] - Y[i];
	}
	sumY += Y[n - 1];
	float scaling = (float(sumX_vec) / 2) / sumY;
	float m0 = sumY_vec / sumX_vec;	
	float sumk = 0.0;
	float mm = sqrt(m0 * m0 + 1);
	int x = 0;
	float y = 0.0;
	Y[0] *= scaling;
	for (size_t i = 1; i < n; i++) {
		Y[i] *= scaling;
		x = X[i] - X[i - 1];
		y = Y[i] - Y[i - 1];
		sumk += -(m0 * x - y) / (x * mm * sqrt(x * x + y * y));
	}
	float r = (1 / (sumk / (n - 1)));
	//std::cout << r;
	int idx1 = 0;
	int idx2 = 1;
	std::vector<int> frontierX = { X[0] };
	std::vector<float> frontierY = { Y[0] };
	point pc = { 0.0,0.0 };
	bool empty = true;
	while (idx2 < n) {
		pc = get_circle(X[idx1], Y[idx1], X[idx2], Y[idx2], r);
		empty = true;
		for (size_t i = idx2 + 1; i < n; i++) {
			if (sqrt(pow(pc.x - X[i], 2) + pow(pc.y - Y[i], 2)) < r) {
				empty = false;
				idx2 += 1;
				break;
			}
		}
		if (empty) {
			frontierX.push_back(X[idx2]);
			frontierY.push_back(Y[idx2]);
			idx1 = idx2;
			idx2 ++;
		}
	}
	for (size_t i = 0; i < frontierY.size(); i++) {
		frontierY[i] /= scaling;
	}
	frontier F = { frontierX, frontierY };
	return F;
}

void write_frontier(frontier V, std::string path = "teste.csv") {
	std::ofstream out(path);
	for (int i = 0; i < V.X.size(); ++i) {
		out << V.X[i] << "," << V.Y[i] << "\n";
	}
	out.close();
}











//bool abs_compare(float a, float b) {
//	return (std::abs(a) < std::abs(b));
//}
//
//int argmax(const std::vector<float>& V, const int x0, const int x1) {
//	auto result = std::max_element(V.begin() + x0, V.end() + x1, abs_compare);
//	int idx = std::distance(V.begin() + x0, result);
//	return idx;
//}
//void write_2d_vector(std::vector<std::vector<float>> V, std::string path = "teste.csv") {
//	std::ofstream out(path);
//	for (int i = 0; i < V.size(); ++i) {
//		out << V[i][0];
//		for (int j = 1; j < V[i].size(); ++j) {
//			out << ',' << V[i][j];
//		}
//		out << "\n";
//	}
//	out.close();
//}
//void write_wav(const std::vector<float> W, std::string path = "test.wav", int fps = 44100) {
//	if (W.size() == 0) {
//		std::cout << "size = 0";
//		return;
//	}
//	const char* fname = path.c_str();
//	SF_INFO sfinfo;
//	sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
//	sfinfo.channels = 1;
//	sfinfo.samplerate = fps;
//
//	SNDFILE* outfile = sf_open(fname, SFM_WRITE, &sfinfo);
//	sf_write_float(outfile, &W[0], W.size());
//	sf_close(outfile);
//	return;
//}
//std::vector<float> interf_trans(const std::vector<float> & W, int res_f = 0, int res_p = 0, float min_f = 0, float min_p = 0, float max_f = 0, float max_p = PI, std::string path = "") {
//	auto tp = Chronograph();
//
//	const int n = W.size();
//	if (res_f == 0) { res_f = (n + 1) / 2; }
//	if (res_p == 0) { res_p = (n + 1) / 2; }
//	if (max_f == 0.0) { max_f = n / 2.0; }
//
//	std::cout << "n=" << n << ", min f=" << min_f << ", max f=" << max_f << ", min p=" << min_p << ", max_p=" << max_p << "\n";
//
//	const int global_res_f = std::round(float(n) / ((max_f - min_f) / float(res_f)));
//
//	const int global_res_p = std::round(2.0 * PI / ((max_p - min_p) / float(res_p)));
//
//	const int res = std::max(global_res_f, global_res_p) + 1;
//
//	const float dp = 2.0 * PI / float(res - 1);
//	const float df = float(n) / float(res - 1);
//
//	std::vector<float> A(res - 1);
//	for (size_t i = 0; i < res - 1; i++) {
//		A[i] = std::cos(float(i) * dp);
//	}
//
//	std::cout << "G res f=" << global_res_f << ", G res p=" << global_res_p << ", G res=" << res << "\n";
//
//	const int f_idx_ini = std::round(min_f / df);
//	const int f_idx_fin = std::round(max_f / df);
//	const int f_step = std::max(1, int(std::round(float(f_idx_fin - f_idx_ini) / float(res_f))));
//
//	const int p_idx_ini = std::round(min_p / dp);
//	const int p_idx_fin = std::round(max_p / dp);
//	const int p_step = std::max(1, int(std::round(float(p_idx_fin - p_idx_ini) / float(res_p))));
//
//	const int rows = std::round(float(f_idx_fin - f_idx_ini) / float(f_step));
//	const int cols = std::round(float(p_idx_fin - p_idx_ini) / float(p_step));
//
//	std::cout << "Rows=" << rows << ", Cols=" << cols << ", f_step=" << f_step << ", p_step=" << p_step << "\n";
//
//	std::vector<std::vector<float>> FP(rows, std::vector<float>(cols));
//	for (auto& i : FP) { std::fill(i.begin(), i.end(), 0); }
//
//	//std::vector<float> scaled_A(res - 1);
//	int idx;
//	for (size_t t = 0; t < n; t++) {
//		//for (size_t i = 0; i < A.size(); i++) {
//		//	scaled_A[i] = A[i] * W[t];
//		//}
//		//const std::vector<float>& scaled_A = mutable_scaled_A;
//		for (size_t i = 0; i < rows; i++) {
//			for (size_t j = 0; j < cols; j++) {
//				idx = ((f_idx_ini + i * f_step) * t + p_idx_ini + j * p_step) % (res - 1);
//				//std::cout << idx << "\n";
//				FP[i][j] += W[t] * A[idx];
//			}
//		}
//	}
//
//	float p = 0.0;
//	float f = 0.0;
//	float a = 0.0;
//	if (path == "") { // won`t save the FP interference matrix
//		for (size_t i = 0; i < rows; i++) {
//			for (size_t j = 0; j < cols; j++) {
//				if (std::abs(FP[i][j]) > a) {
//					a = std::abs(FP[i][j]);
//					f = float(f_idx_ini + (i * f_step)) * df;
//					if (FP[i][j] >= 0.0) {
//						p = float(p_idx_ini + (j * p_step)) * dp;
//					}
//					else {
//						p = float(p_idx_ini + (j * p_step)) * dp + PI;
//					}
//				}
//			}
//		}
//	} else {
//		std::ofstream out(path); // saves FP in path
//		out << "f\\p";
//		for (size_t j = 0; j < cols; j++) {
//			out << ',' << float(p_idx_ini + (j * p_step)) * dp;
//		}
//		out << "\n";
//		for (size_t i = 0; i < rows; i++) {
//			out << float(f_idx_ini + (i * f_step)) * df;
//			for (size_t j = 0; j < cols; j++) {
//				out << ',' << FP[i][j];
//				if (std::abs(FP[i][j]) > a) {
//					a = std::abs(FP[i][j]);
//					f = float(f_idx_ini + (i * f_step)) * df;
//					if (FP[i][j] >= 0.0) {
//						p = float(p_idx_ini + (j * p_step)) * dp;
//					}
//					else {
//						p = float(p_idx_ini + (j * p_step)) * dp + PI;
//					}
//				}
//			}
//			out << "\n";
//		}
//		out.close();
//		std::cout << "saved FP @" << path << "\n";
//	}
//
//	a = a * 2.0 / n;
//	std::cout << "f=" << f << ", p=" << p << ", a=" << a << "\n";
//	tp.stop("Interference Transform Time: ");
//	std::cout << "\n";
//	return { f, p, a };
//}
//
//std::vector<std::vector<float>> interf_trans_n(const std::vector<float>& W, int res_f = 0, int res_p = 0, float min_f = 0, float min_p = 0, float max_f = 0, float max_p = PI) {
//	auto tp = Chronograph();
//
//	const int n = W.size();
//	if (res_f == 0) { res_f = (n + 1) / 2 + 1; }
//	if (res_p == 0) { res_p = (n + 1) / 2; }
//	if (max_f == 0.0) { max_f = n / 2.0; }
//
//	std::cout << "n=" << n << ", min f=" << min_f << ", max f=" << max_f << ", min p=" << min_p << ", max_p=" << max_p << "\n";
//
//	const float df = (max_f - min_f) / float(res_f);
//
//	const float dp = (max_p - min_p) / float(res_p);
//
//	std::cout << "G res f=" << res_f << ", G res p=" << res_p << "\n";
//
//	std::vector<std::vector<float>> FP(res_f, std::vector<float>(res_p));
//	for (auto& i : FP) { std::fill(i.begin(), i.end(), 0.0); }
//
//	float p;
//	float f;
//	for (size_t t = 0; t < n; t++) {
//		for (size_t i = 0; i < res_f; i++) {
//			for (size_t j = 0; j < res_p; j++) {
//				f = min_f + float(i) * df;
//				p = min_p + float(j) * dp;
//				FP[i][j] += std::cos(p + 2 * PI * f * t / n) * W[t];
//			}
//		}
//	}
//
//	std::ofstream out("ITn.csv");
//	out << "f\\p";
//	for (size_t j = 0; j < res_p; j++) {
//		out << ',' << min_p + float(j) * dp;
//	}
//
//	int f_idx = 0;
//	int p_idx = 0;
//	int val = 0;
//
//	out << "\n";
//	for (size_t i = 0; i < res_f; i++) {
//		out << min_f + float(i) * df;
//		for (size_t j = 0; j < res_p; j++) {
//			out << ',' << FP[i][j];
//			if (std::abs(FP[i][j]) > val) {
//				f_idx = i;
//				p_idx = j;
//				val = FP[i][j];
//			}
//		}
//		out << "\n";
//	}
//
//	f = min_f + float(f_idx) * df;
//	p = min_p + float(p_idx) * dp;
//	std::cout << "f=" << f << ", i=" << f_idx << ", p=" << p << ", j=" << p_idx << "\n";
//	tp.stop("Interference Transform Naive Time: ");
//	std::cout << "\n";
//
//	return FP;
//}
//
//std::vector<float> fpa_from_FT(std::vector<std::complex<float>>& FT, std::string path = "") {
//	int n = FT.size();
//	int f;
//	float p;
//	float a;
//	float val = 0.0;
//	float max_val = 0.0;
//	if (path == "") {
//		for (std::size_t i = 0; i < n; ++i) {
//			val = std::pow(FT[i].real(), 2.0) + std::pow(FT[i].imag(), 2.0);
//			if (val > max_val) {
//				max_val = val;
//				f = i;
//				p = std::arg(FT[i]);
//				a = std::abs(FT[i]);
//			}
//		}
//	} else {
//		std::ofstream outfile(path);
//		outfile << "Real, Imag\n";
//		for (std::size_t i = 0; i < n; ++i) {
//			outfile << FT[i].real() << "," << FT[i].imag() << "\n";
//			val = std::pow(FT[i].real(), 2.0) + std::pow(FT[i].imag(), 2.0);
//			if (val > max_val) {
//				max_val = val;
//				f = i;
//				p = std::arg(FT[i]);
//				a = std::abs(FT[i]);
//			}
//		}
//		outfile.close();
//		std::cout << "saved FT @" << path << "\n";
//	}
//	std::cout << "f=" << f << ", p=" << p << ", a=" << a << ", n=" << n << "\n";
//	return { float(f), p, a };
//}
//
//std::vector<float> rfft(std::vector<float>& in) {
//	auto tp = Chronograph();
//	int n = (in.size() + 1) / 2;
//	std::vector<std::complex<float>> out(n);
//
//	DFTI_DESCRIPTOR_HANDLE descriptor;
//	MKL_LONG status;
//
//	status = DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_REAL, 1, in.size()); //Specify size and precision
//	status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE); //Out of place FFT
//	status = DftiCommitDescriptor(descriptor); //Finalize the descriptor
//	status = DftiComputeForward(descriptor, in.data(), out.data()); //Compute the Forward FFT
//	status = DftiFreeDescriptor(&descriptor); //Free the descriptor
//
//	for (size_t i = 0; i < n; i++) {
//		out[i] = out[i] / float(n);
//	}
//
//	auto fpa = fpa_from_FT(out);
//
//	tp.stop("FFT Time: ");
//	std::cout << "\n";
//	return fpa;
//}
//
//std::vector<std::complex<float>> rfft_n(std::vector<float>& in) {
//	auto tp = Chronograph();
//	std::vector<std::complex<float>> out((in.size() + 1) / 2);
//	std::fill(out.begin(), out.end(), 0.0);
//
//	float k;
//	for (size_t f = 0; f < out.size(); f++) {
//		for (size_t t = 0; t < in.size(); t++) {
//			k = 2.0 * PI * f * t / in.size();
//			out[f] += {in[t] * std::cos(k), -in[t] * std::sin(k) };
//		}
//	}
//	int n = out.size();
//	for (size_t i = 0; i < n; i++) {
//		out[i] = out[i] / float(n);
//	}
//
//	fpa_from_FT(out, "FTn.csv");
//
//	tp.stop("FFT naive Time: ");
//	std::cout << "\n";
//	return out;
//}
//
//Wav generate_random_wave(const int n, int seed=0, int number_of_sinusoids=100, bool save=true) {
//
//	std::mt19937 gen(seed); // mersenne_twister_engine
//	std::uniform_int_distribution<int> i_dis(0, n/2); // get index to fill
//	const float bnd = 2.0 / number_of_sinusoids;
//	std::uniform_real_distribution<float> f_dis(- bnd, bnd); // get value to use in index
//
//	std::vector<std::complex<float>>FT(n);
//
//	for (size_t i = 0; i < number_of_sinusoids; i++) {
//		FT[i_dis(gen)] = {f_dis(gen) , f_dis(gen)};
//		//std::cout << i_dis(gen) << " " << f_dis(gen) << f_dis(gen) << "\n";
//	}
//
//	//std::fill(FT.begin(), FT.end(), 0.0);
//	//FT[1] = { 0.0, 0.5 };
//
//	//std::vector<std::complex<float>> out(n);
//
//	DFTI_DESCRIPTOR_HANDLE descriptor;
//	MKL_LONG status;
//	std::vector<float>W_vec(n);
//	status = DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_REAL, 1, FT.size()); //Specify size and precision
//	status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE); //Out of place FFT
//	status = DftiCommitDescriptor(descriptor); //Finalize the descriptor
//	status = DftiComputeBackward(descriptor, FT.data(), W_vec.data()); //Compute the Backward FFT
//	status = DftiFreeDescriptor(&descriptor); //Free the descriptor
//
//	//for (size_t i = 0; i < n; i++) {
//	//	std::cout << W_vec[i] << "\n";
//	//}
//	Wav W(W_vec);
//
//
//	if (save) {
//		W.write();
//	}
//
//	return W;
//}
//
//Wav generate_sinusoid(int n = 1000, float f = 1.0, float p = 0.0, float a = 1.0, bool save=true) {
//
//	std::vector<float> W_vec(n);
//
//	for (size_t x = 0; x < n; x++) {
//		W_vec[x] = a * std::cos(p + 2.0 * PI * f * x / n);
//	}
//
//	Wav W(W_vec);
//
//	if (save) {
//		W.write("sinusoid.wav");
//	}
//	std::cout << "Sinusoid Generated: f=" << f << ", p=" << p << ", a=" << a << ", n=" << n << "\n\n";
//	return W;
//
//}
//
//float error(const std::vector<float>& W1, const std::vector<float>& W2) {
//	int n = W1.size();
//	float e = 0.0;
//	for (size_t i = 0; i < n; i++) {
//		e += std::abs(W1[i] - W2[i]);
//	}
//	return e;
//}