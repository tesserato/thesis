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


	layer(v_inte X, v_real E, v_real W, pint n_) {
		X_PCs = X;
		Envelope = E;
		Waveform = W;
		n = n_;
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

layer compress(const v_real& W, inte mult=3) {
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
		<< "START>> n:" << W.size()
		<< ", Xpcs[-1]:" << best_Xpcs.back()
		<< ", Xpcs size:" << best_Xpcs.size()
		<< ", mode:" << ma.mode
		<< ", abdm:" << ma.abdm
		<< ", min size:" << min_size
		<< ", max size:" << max_size
		<< ", E:" << best_avdv << "\n";
	real avdv;
	v_inte Xpcs;
	v_real avg(best_avg);

	const std::chrono::time_point< std::chrono::steady_clock> start = std::chrono::high_resolution_clock::now();
	real duration = 0.0;
	while (duration <= 50.0) {
		Xpcs = refine_Xpcs(W, avg, min_size, max_size);
		ma = average_pc_waveform(avg, Xpcs, W);
		inte min_size = std::max(inte(10), inte(ma.mode) - inte(mult * ma.abdm));
		inte max_size = ma.mode + inte(mult * ma.abdm);

		Wave_rep = Compressed::raw(Xpcs, avg, W);
		//Wave = Wave_rep.reconstruct_full(fps);
		avdv = error(Wave_rep.W_reconstructed, W);

		std::cout
			//<< "i:" << i
			<< ", n:" << W.size()
			<< ", Xpcs[-1]:" << Xpcs.back()
			<< ", Xpcs size:" << Xpcs.size()
			<< ", mode:" << ma.mode
			<< ", abdm:" << ma.abdm
			<< ", min size:" << min_size
			<< ", max size:" << max_size
			<< ", E:" << avdv << "\n";

		if (avdv < best_avdv) {
			std::cout << "^^^^^IMPROVEMENT!^^^^^\n";
			best_avdv = avdv;
			best_Xpcs = Xpcs;
			best_avg = avg;
		}
		duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
	}

	Wave_rep = Compressed::raw(best_Xpcs, best_avg, W);
	//Wav Wave = Wav(Wave_rep.W_reconstructed, fps);
	//Wave.write(name + "_rec.wav");

	//write_vector(best_Xpcs, name + "_Xpcs_best.csv");
	//write_vector(best_avg, name + "_avgpcw_best.csv");
	//write_vector(Wave_rep.Envelope, name + "_envelope.csv");

	//v_real residue(W.size(), 0.0);
	//for (pint i = 0; i < W.size(); i++) {
	//	residue[i] = W[i] - Wave_rep.W_reconstructed[i];
	//}
	//Wav error = Wav(residue, fps);
	//error.write(name + "_residue.wav");

	//auto Wave_rep_smooth = Compressed::smooth(best_Xpcs, best_avg, Wave_rep.Envelope, W.size());
	//Wav Wave_smooth = Wav(Wave_rep_smooth.W_reconstructed, fps);
	//Wave_smooth.write(name + "_rec_smooth.wav");
	//write_vector(Wave_rep_smooth.Waveform, name + "_avgpcw_best_smooth.csv");
	//write_vector(Wave_rep_smooth.Envelope, name + "_envelope_smooth.csv");
	//time.stop();
	
	return layer(best_Xpcs, Wave_rep.Envelope, best_avg, W.size());
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
int main(int argc, char** argv) {
	std::cout << argc << "\n";
	
	if (argc == 1) {
		for (const auto& entry : std::filesystem::directory_iterator(".")) {
			std::string path = { entry.path().u8string() };		
			if (ends_with(path, ".wav")) {
				std::cout << path << std::endl;
			}
		}
	}
	else {
		for (int i = 1; i < argc; ++i) {
			if (ends_with(argv[i], ".wav")) {
				std::cout << "Compressing " << argv[i] << std::endl;
				auto WV = read_wav(argv[i]);
				auto W = WV.W;
				auto fps = WV.fps;
				auto C = compress(W);
			}
		}
	}
	
	
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

