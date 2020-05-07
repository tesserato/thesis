#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <math.h>
#include <chrono> 
#include <fstream>
#include <algorithm>
#include <sndfile.hh>

const float PI = 3.14159265358979323846;

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

std::vector<std::vector<float>> interf_trans(const std::vector<float> & W, int res_f = 0, int res_p = 0, float min_f = 0, float min_p = 0, float max_f = 0, float max_p = 2 * PI) {
	auto tp = Chronograph();

	const int n = W.size();
	if (res_f == 0) { res_f = (n + 1) / 2; }
	if (res_p == 0) { res_p = (n + 1) / 2; }
	if (max_f == 0.0) { max_f = n / 2.0; }

	std::cout << "n=" << n << ", min f=" << min_f << ", max f=" << max_f << ", min p=" << min_p << ", max_p=" << max_p << "\n";

	const float preliminar_df = (max_f - min_f) / float(res_f);
	const int global_res_f = std::round(float(n) / preliminar_df);

	const float preliminar_dp = (max_p - min_p) / float(res_p);
	const int global_res_p = std::round(2.0 * PI / preliminar_dp);

	const int res = std::max(global_res_f, global_res_p) + 1;

	const float dp = 2.0 * PI / float(res - 1);
	const float df = float(n) / float(res - 1);

	std::vector<float> mutable_A(res - 1);
	for (size_t i = 0; i < res - 1; i++) {
		mutable_A[i] = std::cos(float(i) * dp);
	}
	const std::vector<float>& A = mutable_A;

	std::cout << "G res f=" << global_res_f << ", G res p=" << global_res_p << ", G res=" << res << "\n";

	const int f_idx_ini = std::round(min_f / df);
	const int f_idx_fin = std::round(max_f / df);
	const int f_step = std::max(1, int(std::round(float(f_idx_fin - f_idx_ini) / float(res_f))));

	const int p_idx_ini = std::round(min_p / dp);
	const int p_idx_fin = std::round(max_p / dp);
	const int p_step = std::max(1, int(std::round(float(p_idx_fin - p_idx_ini) / float(res_p))));

	const int rows = std::round(float(f_idx_fin - f_idx_ini) / float(f_step));
	const int cols = std::round(float(p_idx_fin - p_idx_ini) / float(p_step));

	std::cout << "Rows=" << rows << ", Cols=" << cols << ", f step=" << f_step << ", p step=" << p_step << "\n";

	std::vector<std::vector<float>> FP(rows, std::vector<float>(cols));
	for (auto& i : FP) { std::fill(i.begin(), i.end(), 0); }

	std::vector<float> mutable_scaled_A(res - 1);
	int idx;
	for (size_t t = 0; t < n; t++) {
		for (size_t i = 0; i < A.size(); i++) {
			mutable_scaled_A[i] = A[i] * W[t];
		}
		const std::vector<float>& scaled_A = mutable_scaled_A;
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

	const float p = float(p_idx_ini + (p_idx * p_step)) * dp;
	const float f = float(f_idx_ini + (f_idx * f_step)) * df;
	std::cout << "f=" << f << ", i=" << f_idx << ", p=" << p << ", j=" << p_idx << "\n";
	tp.stop("Time: ");

	return FP;
}

std::vector<std::vector<float>> interf_trans_n(const std::vector<float>& W, int res_f = 0, int res_p = 0, float min_f = 0, float min_p = 0, float max_f = 0, float max_p = 2 * PI) {
	auto tp = Chronograph();

	const int n = W.size();
	if (res_f == 0) { res_f = (n + 1) / 2; }
	if (res_p == 0) { res_p = (n + 1) / 2; }
	if (max_f == 0.0) { max_f = n / 2.0; }

	std::cout << "n=" << n << ", min f=" << min_f << ", max f=" << max_f << ", min p=" << min_p << ", max_p=" << max_p << "\n";

	const float df = (max_f - min_f) / float(res_f);

	const float dp = (max_p - min_p) / float(res_p);

	std::cout << "G res f=" << res_f << ", G res p=" << res_p << "\n";


	std::vector<std::vector<float>> FP(res_f, std::vector<float>(res_p));
	for (auto& i : FP) { std::fill(i.begin(), i.end(), 0.0); }

	float p;
	float f;
	for (size_t t = 0; t < n; t++) {
		for (size_t i = 0; i < res_f; i++) {
			for (size_t j = 0; j < res_p; j++) {
				f = min_f + float(i) * df;
				p = min_p + float(j) * dp;
				FP[i][j] += std::cos(p + 2 * PI * f * t / n) * W[t];
			}
		}
	}

	int f_idx = 0;
	int p_idx = 0;
	int val = 0;
	for (size_t i = 0; i < res_f; i++) {
		for (size_t j = 0; j < res_p; j++) {
			if (FP[i][j] > val) {
				f_idx = i;
				p_idx = j;
				val = FP[i][j];
			}
		}
	}

	f = min_f + float(f_idx) * df;
	p = min_p + float(p_idx) * dp;
	std::cout << "f=" << f << ", i=" << f_idx << ", p=" << p << ", j=" << p_idx << "\n";
	tp.stop("Time: ");

	return FP;
}


