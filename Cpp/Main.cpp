#include "Header.h"

int main() {
	//int n = 40000;
	//int t = 5;
	//int res = 40000;

	Wav WA = read_file("04.wav");
	auto W = WA.get_samples();

	auto FP = interference_naive(W, W.size() / 2);

	//std::vector<std::vector<float>> G1 = make_grid(n, t, res);

	//std::vector<std::vector<float>> G2 = make_grid_naive(n, t, res);

	write_2d_vector(FP, "FP.csv");
	//write_2d_vector(G2, "G2.csv");

	//for (size_t i = 0; i < res; i++) {
	//	for (size_t j = 0; j < res; j++) {
	//		if (std::abs(G1[i][j] - G2[i][j]) > 0.1) {
	//			std::cout << "!!! Diff at i=" << i <<" j="<<j ;
	//			return 1;
	//		}
	//	}		
	//}
	//std::cout << "Equal\n";
	return 0;
}

//int main()
//{
//    Wav WA = read_file("test.wav");
//    auto n = WA.get_size();
//    auto X = arma::linspace(0, n - 1, n);
//    float fps = float(WA.get_fps());
//    arma::vec R(n);
//    R.fill(0);
//    auto W = WA.get_normalized_samples();
//
//    /*auto tp = Chronograph();    
//    tp.stop("time: ");*/
//
//    for (size_t ctr = 0; ctr < 1; ctr++) {
//        auto TFPA = grid_naive(W, fps, 50, 1000, 100);
//        write_2d_vector(TFPA);
//        auto Ts = arma::conv_to<arma::vec>::from(TFPA[0]);
//
//        auto As = arma::conv_to<arma::vec>::from(TFPA[3]);
//        As.save("As.csv", arma::csv_ascii);
//        auto Xs = Ts * fps;
//        auto Aa = arma::polyfit(Xs, As, 3);
//        arma::mat A = arma::polyval(Aa, X);
//        A.save("A.csv", arma::csv_ascii);
//
//        auto Fs = arma::conv_to<arma::vec>::from(TFPA[1]);
//        auto Ps = arma::conv_to<arma::vec>::from(TFPA[2]);
//        arma::vec Is = 2 * PI * Fs % Ts + Ps;
//        Is.save("Is.csv", arma::csv_ascii);
//        auto Ai = arma::polyfit(Xs, Is, 3);
//        arma::mat I = arma::polyval(Ai, X);
//        I.save("I.csv", arma::csv_ascii);
//        arma::mat S = A % arma::cos(I);
//        R += S;
//        for (size_t i = 0; i < n; i++) { W[i] -= S[i]; }
//        write_file(arma::conv_to<std::vector<float>>::from(R), "R_" + std::to_string(ctr) + "_.wav");
//        write_file(arma::conv_to<std::vector<float>>::from(S), "S_" + std::to_string(ctr) + "_.wav");
//    }
//    
//
//    //write_2d_vector(TFPA);
//    //auto O = fit(TFPA[0], TFPA[1], W, W.size());
//    //write_file(O, "btst.wav");
//}