#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <math.h> 
//#include <cmath>
#include <chrono> 
#include <fstream>
#include <algorithm>
//#include <complex>
//#define ARMA_DONT_USE_WRAPPER
//#include <armadillo>
#include <sndfile.hh>

const double PI = 3.14159265358979323846;

class Chronograph {
private:
	std::chrono::time_point< std::chrono::steady_clock> start, end;
	int duration;
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
	Wav(std::vector<float> W_, int fps_) {
		fps = fps_;
		W = W_;
		n = W.size();
		a = 0.0;
		for (size_t i = 0; i < n; i++) {
			if (std::abs(W[i]) > a) {
				a = std::abs(W[i]);
			}
		}
		std::cout << "Amplitude: " << a << " n: " << n << " fps: " << fps << "\n";
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

void write_wav(const std::vector<float> W, std::string path = "test.wav", int fps = 44100) {
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
	return;
}

void write_2d_vector(std::vector<std::vector<float>> V, std::string path = "teste.csv") {
	std::ofstream out(path);
	for (int i = 0; i < V.size(); ++i) {
		out << V[i][0];
		for (int j = 1; j < V[i].size(); ++j) {
			out << ',' << V[i][j];
		}
		out << "\n";
	}
}

std::vector<std::vector<float>> interf_trans(std::vector<float> W, int res_f = 0, int res_p = 0, float min_f = 0, float min_p = 0, float max_f = 0, float max_p = 2 * PI) {
	auto tp = Chronograph();

	int n = W.size();
	if (res_f == 0) { res_f = (n + 1) / 2; }
	if (res_p == 0) { res_p = (n + 1) / 2; }
	if (max_f == 0.0) { max_f = n / 2.0; }

	std::cout << "n=" << n << ", min f=" << min_f << ", max f=" << max_f << ", min p=" << min_p << ", max_p=" << max_p << "\n";

	float preliminar_df = (max_f - min_f) / float(res_f);
	int global_res_f = std::round(float(n) / preliminar_df);

	float preliminar_dp = (max_p - min_p) / float(res_p);
	int global_res_p = std::round(2.0 * PI / preliminar_dp);

	int res = std::max(global_res_f, global_res_p);

	float dp = 2.0 * PI / float(res - 1);
	float df = float(n) / float(res - 1);

	std::vector<float>A(res - 1);
	for (size_t i = 0; i < res - 1; i++) {
		A[i] = std::cos(float(i) * dp);
	}

	std::cout << "G res f=" << global_res_f << ", G res p=" << global_res_p << ", G res=" << res << "\n";

	int f_idx_ini = std::round(min_f / df);
	int f_idx_fin = std::round(max_f / df);
	int f_step = std::max(1, int(std::round(float(f_idx_fin - f_idx_ini) / float(res_f))));

	int p_idx_ini = std::round(min_p / dp);
	int p_idx_fin = std::round(max_p / dp);
	int p_step = std::max(1, int(std::round(float(p_idx_fin - p_idx_ini) / float(res_p))));

	int rows = std::round(float(f_idx_fin - f_idx_ini) / float(f_step));
	int	cols = std::round(float(p_idx_fin - p_idx_ini) / float(p_step));

	std::cout << "Rows=" << rows << ", Cols=" << cols << ", f step=" << f_step << ", p step=" << p_step << "\n";

	std::vector<std::vector<float>> FP(rows, std::vector<float>(cols));
	for (auto& i : FP) { std::fill(i.begin(), i.end(), 0); }

	std::vector<float>scaled_A(res - 1);
	int idx;
	for (size_t t = 0; t < n; t++) {
		for (size_t i = 0; i < A.size(); i++) {
			scaled_A[i] = A[i] * W[t];
		}
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				idx = ((f_idx_ini + i * f_step) * t + p_idx_ini + j * p_step) % (res - 1);
				FP[i][j] += scaled_A[idx];
			}
		}
	}

	int f_idx = 0;
	int p_idx = 0;
	int val = 0;
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			if (FP[i][j] > val) {
				f_idx = i;
				p_idx = j;
				val = FP[i][j];
			}
		}
	}

	float p = float(p_idx_ini + (p_idx * p_step)) * dp;
	p = std::fmod(p, 2 * PI);
	float f = float(f_idx_ini + (f_idx * f_step)) * df;
	std::cout << "f=" << f << ", i=" << f_idx << ", p=" << p << ", j=" << p_idx << "\n";
	tp.stop("Time: ");

	return FP;
}


std::vector<std::vector<float>> make_grid_naive(int n, int t, int resolution) {

	auto tp = Chronograph();

	std::vector <float> F(resolution);
	float df = float(n) / float(resolution - 1);
	for (size_t i = 0; i < F.size(); i++) {
		F[i] = float(i) * df;
	}

	std::vector <float> P(resolution);
	float dp = (2 * PI) / float(resolution - 1);
	for (size_t i = 0; i < P.size(); i++) {
		P[i] = float(i) * dp;
	}

	std::vector<std::vector<float>> FP(resolution, std::vector<float>(resolution));
	for (size_t i = 0; i < resolution; i++) {
		for (size_t j = 0; j < resolution; j++) {
			FP[i][j] = std::cos(P[j] + 2.0 * PI * F[i] * float(t) / float(n));
		}
	}

	tp.stop("Naive time: ");

	return FP;
}

std::vector<std::vector<float>> make_grid(int n, int t, int resolution) {

	auto tp = Chronograph();

	//std::vector <float> F(resolution);
	int l = (resolution + 1) / 2;
	std::vector<float> arr(resolution - 1);

	float dp = 2.0 * PI / float(resolution - 1);
	for (size_t i = 0; i < resolution - 1; i++) {
		arr[i] = std::cos(float(i) * dp);
	}

	//for (size_t i = 0; i < arr.size(); i++) {
	//	std::cout << arr[i] << ' ';
	//}

	int bias;
	if (resolution % 2 == 0) {
		bias = 0;
	}
	else {
		bias = 1;
	}

	int idx;
	std::vector<std::vector<float>> FP(resolution, std::vector<float>(resolution));
	for (size_t i = 0; i < l; i++) {
		for (size_t j = 0; j < resolution; j++) {
			idx = (i * t + j) % (resolution - 1);
			FP[i][j] = arr[idx];
			//FP[i][l + j - bias] = arr[idx - l + 1]; //<|<|<|<|<|<|<|<|<|<|
			FP[resolution - i - 1][resolution - j - 1] = arr[idx];
			//FP[resolution - i - 1][resolution - j - l - 1 + bias] = arr[idx - l + 1]; //<|<|<|<|<|<|<|<|<|<|
		}
	}

	tp.stop("Time: ");

	return FP;
}

std::vector<std::vector<float>> interference(std::vector<float> W, int resolution) {
	auto tp = Chronograph();

	int n = W.size();

	int l = (resolution + 1) / 2;

	std::vector<float> sinusoid(resolution - 1);

	float dp = 2.0 * PI / float(resolution - 1);
	for (size_t i = 0; i < resolution - 1; i++) {
		sinusoid[i] = std::cos(float(i) * dp);
	}

	int idx;

	std::vector<std::vector<float>> FP(resolution, std::vector<float>(resolution));
	for (auto& i : FP) { std::fill(i.begin(), i.end(), 0); }

	std::vector<float> A(resolution - 1);
	for (size_t t = 0; t < n; t++) {

		//auto tp_inner = Chronograph();

		for (size_t i = 0; i < resolution - 1; i++) {
			A[i] = sinusoid[i] * W[t];
		}

		for (size_t i = 0; i < l; i++) {
			for (size_t j = 0; j < resolution; j++) {
				idx = (i * t + j) % (resolution - 1);
				FP[i][j] += A[idx];
				FP[resolution - i - 1][resolution - j - 1] += A[idx];
			}
		}
		std::cout << t << " \n";
		//tp_inner.stop("Inner Time: ");
	}
	int f_idx = 0;
	int p_idx = 0;
	int val = 0;
	for (size_t i = 0; i < resolution; i++) {
		for (size_t j = 0; j < resolution; j++) {
			if (FP[i][j] > val) {
				f_idx = i;
				p_idx = j;
				val = FP[i][j];
			}
		}
	}

	float fps = 44100;
	float df = float(n) / float(resolution - 1);
	float f = f_idx * df;
	std::cout << "f=" << f << ", i=" << f_idx << ", p=" << p_idx * dp;
	tp.stop("Time: ");

	return FP;
}

std::vector<std::vector<float>> interference_naive(std::vector<float> W, int resolution) {
	auto tp = Chronograph();

	int n = W.size();

	std::vector <float> F(resolution);
	float df = (float(n) / 2.0) / float(resolution);
	for (size_t i = 0; i < F.size(); i++) {
		F[i] = float(i) * df;
	}

	std::vector <float> P(resolution);
	float dp = (2.0 * PI) / float(resolution);
	for (size_t i = 0; i < P.size(); i++) {
		P[i] = float(i) * dp;
	}

	std::vector<std::vector<float>> FP(resolution, std::vector<float>(resolution));
	for (auto& i : FP) { std::fill(i.begin(), i.end(), 0); }

	for (size_t t = 0; t < n; t++) {
		for (size_t i = 0; i < resolution; i++) {
			for (size_t j = 0; j < resolution; j++) {
				FP[i][j] += (W[t]/* * 2.0 / float(n)*/) * std::cos(P[j] + 2.0 * PI * F[i] * float(t) / float(n));
			}
		}
		std::cout << t << " \n";
		//tp_inner.stop("Inner Time: ");
	}
	int f_idx = 0;
	int p_idx = 0;
	int val = 0;
	for (size_t i = 0; i < resolution; i++) {
		for (size_t j = 0; j < resolution; j++) {
			if (FP[i][j] > val) {
				f_idx = i;
				p_idx = j;
				val = FP[i][j];
			}
		}
	}

	float fps = 44100;
	//float df = float(n) / float(resolution - 1);
	float f = f_idx * df;
	std::cout << "f=" << f << ", i=" << f_idx << ", p=" << p_idx * dp;
	tp.stop("Time: ");

	return FP;
}

std::vector<std::vector<float>> grid_naive(std::vector<float> W, float fps, int steps, int res_f, int res_p, float min_f = NAN, float max_f = NAN) {
	int n = W.size();
	float dt = float(n) / float(steps);

	if (max_f == NAN) {
		max_f = dt / 2.0;
	}
	else {
		max_f = max_f * dt / fps;
	}

	if (min_f == NAN) {
		min_f = 0.0;
	}
	else {
		min_f = min_f * dt / fps;
	}

	float max_p = 2 * PI;

	std::vector <float> F(res_f);
	float df = (max_f - min_f) / float(res_f);
	for (size_t i = 0; i < F.size(); i++) {
		F[i] = min_f + float(i) * df;
	}

	std::vector <float> P(res_p);
	float dp = max_p / float(res_p);
	for (size_t i = 0; i < P.size(); i++) {
		P[i] = float(i) * dp;
	}

	std::cout << "Global df = " << df * float(fps) / float(n) << " | dp = " << dp << "\n";

	std::vector<std::vector<float>> FP(res_f, std::vector<float>(res_p));

	std::vector<std::vector<float>> TFPA(4, std::vector<float>(steps));

	//float total = 0;	
	int f_idx = 0;
	int p_idx = 0;
	float val = 0;
	float neg = 0.0;

	for (size_t s = 1; s < steps + 1; s++) {
		std::cout << "step " << s << " of " << steps << "\n";
		for (auto& i : FP) { std::fill(i.begin(), i.end(), 0); }

		for (size_t t = dt * (s - 1); t < dt * s; t++) {
			for (size_t i = 0; i < res_f; i++) {
				for (size_t j = 0; j < res_p; j++) {
					FP[i][j] += (W[t] * 2.0 / dt) * std::cos(P[j] + 2.0 * PI * F[i] * float(t) / float(n));
				}
			}
		}

		f_idx = 0;
		p_idx = 0;
		val = 0;
		for (size_t i = 0; i < res_f; i++) {
			for (size_t j = 0; j < res_p; j++) {
				if (std::abs(FP[i][j]) > val) {
					if (FP[i][j] > 0) {
						f_idx = i;
						p_idx = j;
						val = FP[i][j];
						neg = 0.0;
					}
					else {
						f_idx = i;
						p_idx = j;
						val = -FP[i][j];
						neg = 1.0;
					}
				}
			}
		}
		TFPA[0][s - 1] = (dt * float(s) - dt / 2.0) / fps;
		TFPA[1][s - 1] = F[f_idx] * fps / float(n);
		TFPA[2][s - 1] = P[p_idx] + neg * PI;
		TFPA[3][s - 1] = val;
		std::cout << "t = " << TFPA[0][s - 1] << " | f = " << TFPA[1][s - 1] << " | p = " << TFPA[2][s - 1] << " | a = " << TFPA[3][s - 1] << "\n";
	}
	return TFPA;
}


