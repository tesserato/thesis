#include "Header.h"
#include <filesystem>


template <typename T> void write_vector(std::vector<T>& V, std::string path = "teste.csv") {
	std::ofstream out(path);
	for (pint i = 0; i < V.size() - 1; ++i) {
		out << V[i] << ",";
	}
	out << V.back();
	out.close();
}



int main() {
	Chronograph time;

	std::string name = "amazing";
	auto WV = read_wav(name + ".wav");
	auto W = WV.W;
	auto fps = WV.fps;
	//WV.write();

	v_pint posX, negX;
	get_pulses(W, posX, negX);


	auto posF = get_frontier(W, posX);
	auto negF = get_frontier(W, negX);
	write_vector(posF, name + "_pos.csv");
	write_vector(negF, name + "_neg.csv");


	refine_frontier_iter(posF, W);
	refine_frontier_iter(negF, W);
	write_vector(posF, name + "_pos_n.csv");
	write_vector(negF, name + "_neg_n.csv");

	inte mult{ 3 };
	v_inte best_Xpcs = get_Xpcs(posF, negF);
	write_vector(best_Xpcs, name + "_Xpcs.csv");
	v_real best_avg;
	mode_abdm ma = average_pc_waveform(best_avg, best_Xpcs, W);
	write_vector(best_avg, name + "_avgpcw.csv");

	Compressed Wave_rep = Compressed(best_Xpcs, best_avg, W);
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

	for (pint i = 0; i < 500; i++) {
		Xpcs = refine_Xpcs(W, avg, min_size, max_size);
		ma = average_pc_waveform(avg, Xpcs, W);
		inte min_size = std::max(inte(10), inte(ma.mode) - inte(mult * ma.abdm));
		inte max_size = ma.mode + inte(mult * ma.abdm);

		Wave_rep = Compressed(Xpcs, avg, W);
		//Wave = Wave_rep.reconstruct_full(fps);
		real avdv = error(Wave_rep.W_reconstructed, W);

		std::cout
			<< "i:" << i
			<< ", n:" << W.size()
			<< ", Xpcs[-1]:" << Xpcs.back()
			<< ", Xpcs size:" << Xpcs.size()
			<< ", mode:" << ma.mode
			<< ", abdm:" << ma.abdm
			<< ", min size:" << min_size
			<< ", max size:" << max_size
			<< ", E:" << avdv << "\n";

		if (avdv < best_avdv) {
			std::cout  << "^^^^^IMPROVEMENT!^^^^^\n";
			best_avdv = avdv;
			best_Xpcs = Xpcs;
			best_avg = avg;
		}
	}

	Wave_rep = Compressed(best_Xpcs, best_avg, W);
	Wav Wave = Wav(Wave_rep.W_reconstructed, fps);
	Wave.write(name + "_rec.wav");
	write_vector(best_Xpcs, name + "_Xpcs_best.csv");
	write_vector(best_avg, name + "_avgpcw_best.csv");

	time.stop();
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

