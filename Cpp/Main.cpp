#include "Header.h"
#include <filesystem>

class layer {
private:
	v_real W;
	bool reconstructed = false;
public:
	v_inte X_PCs; // start of each pulse
	v_real Envelope;
	v_real Waveform;
	pint n;
	pint fps;


	layer(v_inte X, v_real E, v_real W, pint n_, pint fps_) {
		X_PCs = X;
		Envelope = E;
		Waveform = W;
		n = n_;
		fps = fps_;
	}
	pint size() {
		return X_PCs.size() + Envelope.size() + Waveform.size();
	}
	v_real reconstruct() {
		if (reconstructed) {
			return W;
		}
		boost::math::interpolators::cardinal_cubic_b_spline<real> spline(Waveform.begin(), Waveform.end(), 0.0, 1.0 / real(Waveform.size()));
		v_real W_reconstructed(n, 0.0);

		inte x0{ X_PCs[0] };
		inte x1{ X_PCs[1] };
		real step{ 1.0 / (x1 - x0) };
		for (pint i = 1; i < X_PCs.size() - 1; i++) {
			x0 = X_PCs[i];
			x1 = X_PCs[i + 1];
			step = 1.0 / real(x1 - x0);
			for (inte j = x0; j < x1; j++) {
				W_reconstructed[j] = spline((j - x0) * step) * Envelope[i];
			}
			if (x1 - x0 <= 3) {
				std::cout << "Warning: Pseudo cycle with period < 4 between " << x0 << " and " << x1 << "\n";
			}
		}

		inte X_PCs_start = X_PCs[1];
		inte X_PCs_end = X_PCs.back();
		pint itens{ 5 };
		real avg_t{ 0.0 };
		real avg_e{ 0.0 };
		while (X_PCs[0] > 0) {
			avg_t = 0.0;
			avg_e = 0.0;
			for (pint i = 0; i < itens; i++) {
				avg_t += X_PCs[i + 1] - X_PCs[i];
				avg_e += Envelope[i];
			}
			X_PCs.insert(X_PCs.begin(), X_PCs[0] - std::round(avg_t / itens));
			Envelope.insert(Envelope.begin(), avg_e / itens);
		}

		while (X_PCs.back() < n) {
			avg_t = 0.0;
			avg_e = 0.0;
			for (pint i = X_PCs.size() - itens - 1; i < X_PCs.size() - 1; i++) {
				avg_t += X_PCs[i + 1] - X_PCs[i];
				avg_e += Envelope[i];
			}
			X_PCs.push_back(X_PCs.back() + std::round(avg_t / itens));
			Envelope.push_back(avg_e / itens);
		}

		// Filling from 0 to Xp[1]
		pint ctr = 0;
		x0 = X_PCs[ctr];
		x1 = X_PCs[ctr + 1];
		while (x1 <= X_PCs_start) {
			if (x1 < 0) {
				continue;
			}
			else {
				step = 1.0 / real(x1 - x0);
				for (inte j = x0; j < x1; j++) {
					if (j >= 0) {
						W_reconstructed[j] = spline((j - x0) * step) * Envelope[ctr];
					}
				}
			}
			ctr++;
			x0 = X_PCs[ctr];
			x1 = X_PCs[ctr + 1];
		}

		ctr = X_PCs.size() - 1;
		x0 = X_PCs[ctr - 1];
		x1 = X_PCs[ctr];
		while (x0 >= X_PCs_end) {
			if (x0 >= n) {
				continue;
			}
			else {
				step = 1.0 / real(x1 - x0);
				for (inte j = x0; j < x1; j++) {
					if (j < n) {
						W_reconstructed[j] = spline((j - x0) * step) * Envelope.back();
					}
				}
			}
			ctr--;
			x0 = X_PCs[ctr - 1];
			x1 = X_PCs[ctr];
		}
		reconstructed = true;
		W = W_reconstructed;
		return W_reconstructed;
	}
};

Compressed compress(const v_real& W, inte max_seconds = 100, pint max_iterations = std::numeric_limits<pint>::max(), inte mult = 3) {
	v_pint posX, negX;
	get_pulses(W, posX, negX);

	auto posF = get_frontier(W, posX);
	auto negF = get_frontier(W, negX);
	//write_vector(posF, name + "_pos.csv");
	//write_vector(negF, name + "_neg.csv");


	refine_frontier_iter(posF, W);
	refine_frontier_iter(negF, W);
	//write_vector(posF, name + "_pos_refined.csv");
	//write_vector(negF, name + "_neg_refined.csv");

	//inte mult{ 3 };
	v_inte best_Xpcs = get_Xpcs(posF, negF);
	//write_vector(best_Xpcs, name + "_Xpcs.csv");
	v_real best_avg;
	mode_abdm ma = average_pc_waveform(best_avg, best_Xpcs, W);
	//write_vector(best_avg, name + "_avgpcw.csv");

	Compressed Wave_rep = Compressed::raw(best_Xpcs, best_avg, W);
	//Wav Wave = Wave_rep.reconstruct_full(fps);
	real best_avdv = error(Wave_rep.W_reconstructed, W);// average_pc_metric(best_avg, best_Xpcs, W);
	inte min_size = std::max(inte(10), inte(ma.mode) - inte(mult * ma.abdm));
	inte max_size = ma.mode + inte(mult * ma.abdm);

	std::cout
		<< "\n"
		<< "Number of samples in the original signal:" << W.size() << "\n"
		<< "Appr. compressed size:" << best_Xpcs.size() + best_avg.size() << "\n"
		<< "Xpcs size:            " << best_Xpcs.size() << "\n"
		<< "Waveform length mode: " << ma.mode << "\n"
		<< "Waveform length abdm: " << ma.abdm << "\n"
		<< "Min waveform length:  " << min_size << "\n"
		<< "Max waveform length:  " << max_size << "\n"
		<< "Initial average error:" << best_avdv << "\n";

	real avdv;
	v_inte Xpcs;
	v_real avg(best_avg);

	const std::chrono::time_point< std::chrono::steady_clock> start = std::chrono::high_resolution_clock::now();
	real duration = 0.0;
	pint ctr = 0;
	while (duration <= max_seconds && ctr < max_iterations) {
		Xpcs = refine_Xpcs(W, avg, min_size, max_size);
		ma = average_pc_waveform(avg, Xpcs, W);
		inte min_size = std::max(inte(10), inte(ma.mode) - inte(mult * ma.abdm));
		inte max_size = ma.mode + inte(mult * ma.abdm);

		Wave_rep = Compressed::raw(Xpcs, avg, W);
		//Wave = Wave_rep.reconstruct_full(fps);
		avdv = error(Wave_rep.W_reconstructed, W);

		if (avdv < best_avdv) {
			best_avdv = avdv;
			best_Xpcs = Xpcs;
			best_avg = avg;

			pint cn = 2 * (best_Xpcs.size() + best_avg.size()) + 1;
			std::cout
				<< "\n"
				<< "Iteration " << ctr << "\n"
				<< "Number of samples in the original signal:" << W.size() << "\n"
				//<< ", Xpcs[-1]:" << best_Xpcs.back()
				<< "Appr. compressed size:" << cn << " (" <<  float(cn) / float(W.size()) << ")\n"
				<< "Waveform length mode: " << ma.mode << "\n"
				<< "Waveform length abdm: " << ma.abdm << "\n"
				<< "Min waveform length:  " << min_size << "\n"
				<< "Max waveform length:  " << max_size << "\n"
				<< "Average error:        " << best_avdv << "\n";
		}
		ctr++;
		duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
	}

	return Compressed::raw(best_Xpcs, best_avg, W);
}

v_real get_residue(const v_real& W0, const v_real& W1) {
	v_real R(W0.size(), 0.0);
	for (pint i = 0; i < R.size(); i++)	{
		R[i] = W0[i] - W1[i];
	}
	return R;
}

inline bool ends_with(std::string const& value, std::string const& ending) {
	if (ending.size() > value.size()) return false;
	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void write_bin(std::string path, pint orig_n, pint fps, const v_inte& beg_of_pseudo_cycles, const v_real& waveform, const v_real& amp_of_pseudo_cycles) {

	std::ofstream data_file;      // pay attention here! ofstream

	v_pint pint_data = { orig_n, fps, amp_of_pseudo_cycles.size(), waveform.size() }; // header

	data_file.open(path, std::ios::out | std::ios::binary | std::fstream::trunc);
	data_file.write((char*) &pint_data[0], pint_data.size() * sizeof(pint));
	data_file.close();

	data_file.open(path, std::ios::out | std::ios::binary | std::fstream::app);
	data_file.write((char*)&beg_of_pseudo_cycles[0], beg_of_pseudo_cycles.size() * sizeof(inte));
	data_file.close();

	data_file.open(path, std::ios::out | std::ios::binary | std::fstream::app);
	data_file.write((char*)&waveform[0], waveform.size() * sizeof(real));
	data_file.close();

	data_file.open(path, std::ios::out | std::ios::binary | std::fstream::app);
	data_file.write((char*)&amp_of_pseudo_cycles[0], amp_of_pseudo_cycles.size() * sizeof(real));
	data_file.close();
}

layer read_bin(std::string path) {
	std::ifstream  data_file;
	data_file.open(path, std::ios::in | std::ios::binary);

	pint* header = new pint[4];
	data_file.read(reinterpret_cast<char*>(&header[0]), 4 * sizeof(pint));
	for (int i = 0; i < 4; ++i) {
		std::cout << header[i] << "\n";
	}

	inte* beg_of_pseudo_cycles_buffer = new inte[header[2] + 1];
	data_file.read((char*)&beg_of_pseudo_cycles_buffer[0], (header[2] + 1) * sizeof(inte));
	v_inte beg_of_pseudo_cycles(beg_of_pseudo_cycles_buffer, beg_of_pseudo_cycles_buffer + header[2] + 1);

	real* waveform_buffer = new real[header[3]];
	data_file.read((char*)&waveform_buffer[0], (header[3]) * sizeof(real));
	v_real waveform(waveform_buffer, waveform_buffer + header[3]);

	real* envelope_buffer = new real[header[2]];
	data_file.read((char*)&envelope_buffer[0], (header[2]) * sizeof(real));
	v_real envelope(envelope_buffer, envelope_buffer + header[2]);

	data_file.close();

	return layer(beg_of_pseudo_cycles, envelope, waveform, header[0], header[1]);
}

int main(int argc, char** argv) {
	pint max_secs = 100;
	pint max_iters = std::numeric_limits<pint>::max();
	std::vector<std::string> wav_paths;
	std::vector<std::string> cmp_paths;
	std::string append = "reconstructed";
	for (int i = 1; i < argc; ++i) { // parsing args
		if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
			std::cout
				<< " For more info about the author: carlos-tarjano.web.app\n"
				<< " Usage: \n"
				<< " [-t x] [-i y] -[a z] [path/to/file1.wav]...[path/to/filen.wav]  [path/to/file1.cmp]...[path/to/filem.cmp]\n"
				<< " -t or --time: (default " << max_secs <<" s) maximum time in seconds allowed for each compression task\n"
				<< " -i or --iterations: maximum number of iterations allowed for each compression task\n"
				<< " -a or --append: (default \"" << append << "\") string to be appended to each reconstructed file name\n"
				<< " If no path is given the root folder will be scanned for .wav and .cmp files, and those will be processed accordingly\n";
			return 0;
		}
		else if (std::strcmp(argv[i], "-t") == 0 || std::strcmp(argv[i], "--time") == 0) {
			max_secs = strtol(argv[i + 1], nullptr, 0);
			std::cout << "Maximum time allowed: "<< max_secs << " seconds\n";
			i++;
		}
		else if (std::strcmp(argv[i], "-i") == 0 || std::strcmp(argv[i], "--iterations") == 0) {
			max_iters = strtol(argv[i + 1], nullptr, 0);
			std::cout << "Maximum iterations allowed: " << max_iters << " \n";
			i++;
		}
		else if (std::strcmp(argv[i], "-a") == 0 || std::strcmp(argv[i], "--append") == 0) {
			append = argv[i + 1];
			std::cout << "Append \"" << append << "\" to the name of reconstructed files\n";
			i++;
		}
		else if (ends_with(argv[i], ".wav")) {
			wav_paths.push_back(argv[i]);
		}
		else if (ends_with(argv[i], ".cmp")) {
			cmp_paths.push_back(argv[i]);
		}
	}
	if (wav_paths.empty() && cmp_paths.empty()) { // no files found in args, searching root
		for (const auto& entry : std::filesystem::directory_iterator("./")) {
			std::string path = { entry.path().u8string() };
			if (ends_with(path, ".wav")) {
				wav_paths.push_back(path);
			}
			else if (ends_with(path, ".cmp")) {
				cmp_paths.push_back(path);
			}
		}
	}

	for (auto path : wav_paths) {
		std::cout << "\nCompressing " << path << std::endl;
		auto WV = read_wav(path);
		auto W = WV.W;
		auto fps = WV.fps;
		auto C = compress(W, max_secs, max_iters);

		path.replace(path.end() - 4, path.end(), ".cmp");
		write_bin(path, W.size(), WV.fps, C.X_PCs, C.Waveform, C.Envelope);
	}

	for (auto path : cmp_paths) {
		std::cout << "\nDecompressing " << path << std::endl;

		auto rec = read_bin(path);
		Wav WW = Wav(rec.reconstruct(), rec.fps);
		path.replace(path.end() - 4, path.end(), "_" + append + ".wav");
		WW.write(path);
	}

	//std::cout << argc << "\n";
	//
	//if (argc == 1) {
	//	for (const auto& entry : std::filesystem::directory_iterator(".")) {
	//		std::string path = { entry.path().u8string() };		
	//		if (ends_with(path, ".wav")) {
	//			std::cout << path << std::endl;
	//		}
	//	}
	//}
	//else {
	//	for (int i = 1; i < argc; ++i) {
	//		if (ends_with(argv[i], ".wav")) {
	//			std::cout << "Compressing " << argv[i] << std::endl;
	//			auto WV = read_wav(argv[i]);
	//			auto W = WV.W;
	//			auto fps = WV.fps;
	//			auto C = compress(W, 5);

	//			std::string path(argv[i]);
	//			path.replace(path.end() - 4, path.end(), ".cmp");
	//			write_bin(path, W.size(), WV.fps, C.X_PCs, C.Waveform, C.Envelope);
	//		}

	//		if (ends_with(argv[i], ".cmp")) {
	//			std::cout << "Decompressing " << argv[i] << std::endl;

	//			auto rec = read_bin(argv[i]);
	//			Wav WW = Wav(rec.reconstruct(), rec.fps);

	//			std::string path(argv[i]);
	//			path.replace(path.end() - 4, path.end(), "_reconstructed.wav");
	//			WW.write(path);
	//		}
	//	}
	//}
	//



	
	//std::string name = "alto";
	//auto WV = read_wav(name + ".wav");
	//auto W = WV.W;
	//auto fps = WV.fps;

	//auto l1 = compress(W);
	//std::cout << l1.size() << "\n";

	//Wav rec1 = Wav(l1.reconstruct(), fps);
	//rec1.write(name + "_reconstructed_01.wav");


	//auto r1 = get_residue(W, l1.reconstruct());


	//auto l2 = compress(r1);
	//Wav rec2 = Wav(l2.reconstruct(), fps);
	//rec2.write(name + "_reconstructed_02.wav");



	//WV.write();

}

//int main() {
//	std::string name = "amazing";
//	auto W = read_wav(name + ".wav").W;
//	//for (pos_integer i = 500; i < 800; i++) {
//	//	W[i] = 0;
//	//}
//
//	//for (pos_integer i = 1000; i < 1700; i++) {
//	//	W[i] = 0;
//	//}
//
//	std::vector<pos_integer> Xz = find_zeroes(W);
//	write_vector(Xz, name + "_zeroes.csv");
//
//	Wav wav(W);
//	wav.write(name + "_alt.csv");
//}

//int main(int argc, char* argv[]) {
//	//std::cout << argc << "\n";
//	if (argc > 1) {
//		std::cout << "has args\n";
//		Chronograph time;
//		for (pos_integer i = 1; i < argc; i++) {
//			std::cout << i << ": " << argv[i] << "\n";
//			//frontier_from_wav(argv[i]);
//		}
//		time.stop();
//	}
//	else {
//		std::cout << "no args " << argv[0] << "\n";
//		Chronograph time;
//		std::string path;
//		for (auto& p : std::filesystem::recursive_directory_iterator("./")) {
//			if (p.path().extension() == ".wav") {
//				path = p.path().stem().string() + ".wav";
//				std::cout << path << '\n';
//				//frontier_from_wav(path);
//			}
//		}
//		time.stop();
//	}
//	return 0;
//}

