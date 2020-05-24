#include "Header.h"

int main() {

	Wav WA = read_wav("Wavs/local_f=2-p=0-n=1000.wav");
	auto W = WA.get_samples();

	auto IT = interf_trans(W, 0, 2);

	auto ITn = interf_trans_n(W, 0, 2);

	auto FT = rfft(W);

	auto FTn = rfft_n(W);

	char dummy;
	std::cin.get(dummy);

	return 0;
}

