#include "Header.h"

int main() {

	//std::vector<float> test = { 0,1,2,3,4,-15,6,7,8 };
	//int idx = argmax(test, 0, 8);
	//std::cout << idx;

	Chronograph time;

	Wav W = read_wav("piano33.wav");

	std::vector<int> posX;
	std::vector<float> posY;
	std::vector<int> negX;
	std::vector<float> negY;

	get_pulses(W.get_samples(), posX, posY, negX, negY);

	frontier posf = get_frontier(posX, posY);
	frontier negf = get_frontier(negX, negY);

	time.stop();

	write_frontier(posf, "pos.csv");
	write_frontier(negf, "neg.csv");





	//int n = 100;

	//Wav WA = generate_sinusoid(n, 2.4);

	//auto W = WA.get_samples();

	//auto fpa_IT = interf_trans(W, n, 2);
	//auto W_IT = generate_sinusoid(n, fpa_IT[0], fpa_IT[1], fpa_IT[2]);
	//auto e_IT = error(W, W_IT.get_samples());

	//auto fpa_FT = rfft(W);
	//auto W_FT = generate_sinusoid(n, fpa_FT[0], fpa_FT[1], fpa_FT[2]);
	//auto e_FT = error(W, W_FT.get_samples());

	//std::cout << "e_IT=" << e_IT << " | e_FT=" << e_FT << "\n";

	//Wav WA = read_wav("Wavs/local_f=2-p=0-n=1000.wav");
	//auto ITn = interf_trans_n(W, 0, 4);
	//auto FTn = rfft_n(W);
	//char dummy;
	//std::cin.get(dummy);

	return 0;
}

