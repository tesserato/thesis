#include "Header.h"
#include <filesystem>


template <typename T> void write_vector(std::vector<T>& V, std::string path = "teste.csv") {
	std::ofstream out(path);
	for (size_t i = 0; i < V.size() - 1; ++i) {
		out << V[i] << ",";
	}
	out << V.back();
	out.close();
}



int main() {
	Chronograph time;


	std::string name = "piano33";
	auto W = read_wav(name + ".wav").W;


	std::vector<size_t> posX, negX;
	get_pulses(W, posX, negX);


	auto posF = get_frontier(W, posX);
	auto negF = get_frontier(W, negX);
	write_vector(posF, name + "_pos.csv");
	write_vector(negF, name + "_neg.csv");


	refine_frontier_iter(posF, W);
	refine_frontier_iter(negF, W);
	write_vector(posF, name + "_pos_n.csv");
	write_vector(negF, name + "_neg_n.csv");

	int mult{ 2 };

	std::vector<double> best_avg;
	auto best_Xpcs = get_Xpcs(posF, negF);
	write_vector(best_Xpcs, name + "_Xpcs.csv");
	mode_abdm ma = average_pc_waveform(best_avg, best_Xpcs, W);
	write_vector(best_avg, name + "_avgpcw.csv");
	double best_avdv = average_pc_metric(best_avg, best_Xpcs, W);
	int min_size = std::max(10, int(ma.mode) - int(mult * ma.abdm));
	int max_size = ma.mode + int(mult * ma.abdm);

	std::cout
		<< "START>> n:" << W.size()
		<< ", Xpcs[-1]:" << best_Xpcs.back()
		<< ", Xpcs size:" << best_Xpcs.size()
		<< ", mode:" << ma.mode
		<< ", abdm:" << ma.abdm
		<< ", min size:" << min_size
		<< ", max size:" << max_size
		<< ", E:" << best_avdv << "\n";

	double avdv;
	std::vector<size_t> Xpcs;
	std::vector<double> avg(best_avg);

	for (size_t i = 0; i < 200; i++) {
		Xpcs = refine_Xpcs(W, avg, min_size, max_size);
		ma = average_pc_waveform(avg, Xpcs, W);
		int min_size = std::max(10, int(ma.mode) - int(mult * ma.abdm));
		int max_size = ma.mode + int(mult * ma.abdm);
		avdv = average_pc_metric(avg, Xpcs, W);

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

	Compressed Wave_rep = Compressed(best_Xpcs, best_avg, W);
	Wav Wave = Wave_rep.reconstruct();
	Wave.write(name + "_rec.wav");
	write_vector(best_Xpcs, name + "_Xpcs_best.csv");
	write_vector(best_avg, name + "_avgpcw_best.csv");

	time.stop();
}

//int main() {
//	std::string name = "tom";
//	auto W = read_wav(name + ".wav").W;
//	for (size_t i = 500; i < 800; i++) {
//		W[i] = 0;
//	}
//
//	for (size_t i = 1000; i < 1700; i++) {
//		W[i] = 0;
//	}
//
//	std::vector<size_t> Xz = find_zeroes(W);
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
//		for (size_t i = 1; i < argc; i++) {
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

