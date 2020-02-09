# Reverb algorithms prototyped with NumPy and SciPy
Scripts implement three examples of reverberation algorithms, based on scientific works on the subject, mainly:

* Moorer J. A., About This Reverberation Business, Computer Music Journal Vol. 3, No. 2, July 1979, p. 13 – 28.
* Jot J. – M., Chaigne A., Digital Delay Networks for Designing Artificial Reverberators, 90th Convetion of Audio Engineering Society, February 1991, preprint 3030.
* Gardner W. G., Reverberation Algorithms, In: Kahrs M., Brandenburg K. (eds) Applications of Digital Signal Processing to Audio and Acoustics. The International Series in Engineering and Computer Science, vol 437. Springer, Boston, MA 2002.

The algorithms are:
* Convolution reverb, requiring an input file and a file containing an impulse response.
* Reverb based on allpass and comb filters, implemented in accordance with Moorer's work and Freeverb's stereo separation method (1979).
* Reverb based on a feedback delay network, implemented in accordance with Jot's and Chaigne's work (1991).

Codes operate on 44.1 kHz 16-bit stereo WAV files and provide a functionality to write a final stereo signal to the file with the same parameters.
The final signals are visualized with matplotlib library.

The scripts were created as a part of an engineering thesis at AGH University of Science and Technology in Cracow, Poland.
