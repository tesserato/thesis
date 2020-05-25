#include "Header.h"

int main() {

	int n = 1000;

	Wav WA = generate_sinusoid(n, 2.4);

	auto W = WA.get_samples();

	auto fpa_IT = interf_trans(W, n, 2);
	auto W_IT = generate_sinusoid(n, fpa_IT[0], fpa_IT[1], fpa_IT[2]);
	auto e_IT = error(W, W_IT.get_samples());

	auto fpa_FT = rfft(W);
	auto W_FT = generate_sinusoid(n, fpa_FT[0], fpa_FT[1], fpa_FT[2]);
	auto e_FT = error(W, W_FT.get_samples());

	std::cout << "e_IT=" << e_IT << " | e_FT=" << e_FT << "\n";

	//Wav WA = read_wav("Wavs/local_f=2-p=0-n=1000.wav");
	//auto ITn = interf_trans_n(W, 0, 4);
	//auto FTn = rfft_n(W);
	//char dummy;
	//std::cin.get(dummy);

	return 0;
}

