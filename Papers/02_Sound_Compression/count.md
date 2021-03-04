Many of the proposed benefits of an electronic drumkit, like portability and practicity for example, are compromised when, to obtain a realistic sound on par with its acoustic counterpart, the output of the drumkit has to be processed in another piece of hardware, such as a personal computer.

The same is true for digital pianos, where models providing realistic sounds are many times more expensive than entry level models, owing in great part to the additional storage needed.

This transforms the instrument in a mere controller or practice aid, robbing it of characteristics that provide the personality of the conventional instruments, such as a particular timbre.

Domain specific compression algorithms and codecs exist in many areas. \textcite{2016ChenCompressing} for example, presents a scheme for compressing convolutional neural networks, while \textcite{2014CanovasLossy}, besides revising existing methods, proposes two lossy schemes for the compression of quality score data of genomic sequencing. 

More recently, the work of \textcite{2019CalhounExploring}, explores the benefits of specific lossy compression algorithms in the context of checkpoints of computational simulations that are stored between simulation sessions.

Regarding sample based digital musical instruments, to the best of the authors' knowledge, no such specialized algorithms exist. Those signals, being highly uniform, offer an opportunity for compression that is not present in most general sounds and remain largely unexplored.

This tendency arises in part from the marketing strategy of the vast majority of those products, where the sheer size of the library is advertised and seems to be sold as an equivalent of quality and versatility. An overall prejudice against compressed formats in general from the part of the music productions community supports this distorted view.

% Literature review | Brief Context of Prior Research
In that scenario, the work of \textcite{2019BlauRethinking} and, more recently, \textcite{2020O’GradyRethinking} are paramount in shedding light to the potential misconceptions permeating the opinion about lossy codecs, albeit from very different standpoints.

The first authors argue, from a technical point of view that perceptual quality metrics should be favored over metrics naively based on the mathematical aspect of Shannon’s rate-distortion theory \parencite{Shannon_1948}, the most popular metrics currently used in the assessment of lossy compression codecs. 

The second work analyses the problem from a somewhat political viewpoint, arguing that at last some of the criticism directed to the MP3 standard and lossy codecs in general is meant to stablish a symbolic capital in the face of the crescent struggle between established recording labels and alternative channel of music diffusion, such as streaming services.

% put your research in the context of other research
Nevertheless, research in data compression in general continues with recent efforts in reviewing the advancements of the field in general presented in \textcite{2021Jayasankarsurvey}. Account of the research in more specific areas, such as video \parencite{2019LaudeComprehensive} and image \parencite{2014RehmanImage} were also produced recently.

Concerning sound lossles audio compression research seems to be lagging when compared to lossy approaches; \textcite{2001HansLossless} suggested that a limit was about to be reached.

New technologies such as 3D, immersive audio has propelled specific compression algorithms, such as the one presented in \textcite{2021HuAudio}. The renaissance of neural networks led to interesting results, as presented in 


The ACER codec, as presented in \textcite{2014CunninghamData} offers an unconventional approach to compression, in that it identifies redundant pieces in a signal that are indexed in a dictionary. The results seems to be satisfactory from a perceptual standpoint, in comparison to MP3 and AAC codecs \parencite{2019CunninghamSubjective}. 