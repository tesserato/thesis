#include "Header.h"

int main() {

	Wav WA = read_wav("Wavs/local_f=2.wav");
	auto W = WA.get_samples();

	auto FP = interf_trans(W, 2000, 500);

	write_2d_vector(FP, "FP.csv");

	char dummy;
	std::cin.get(dummy);

	return 0;
}

