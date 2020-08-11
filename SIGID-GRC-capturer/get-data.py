from rtlsdr import RtlSdr
from pylab import *
from rtlsdr import *
import numpy
import struct


def capture(freq, label, range_freq, range_gain):
	for freq_var in range_freq:
		for gain_var in range_gain:
			# Configure device (freq in herz, freq_correction is PPM)
			sdr = RtlSdr()
			sdr.sample_rate = 250000
			sdr.center_freq = freq + freq_var
			sdr.freq_correction = 60
			sdr.gain = gain_var
			# Read 10 seconds of 250k sampled data
			samples = sdr.read_samples(10 * 256 * 1024)
			sdr.close()

			# Use matplotlib to estimate and plot the PSD
			psd(samples[:65536], NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
			#print(psd)
			xlabel('Frequency (MHz)')
			ylabel('Relative power (dB)')
			# Show and save a pic
			#show()
			file_name = 'data-' + label + '-' + str(double(freq + freq_var)) + '-g-' + str(gain_var)
			print("Saving " + file_name)
			savefig(file_name + '.png')

			# Save samples to file
			fh = open(file_name + '.iq', "wb")
			x = samples
			x = x[:5000000]

			for sample in x:
				ba = bytearray(struct.pack("f", numpy.float32(sample.real)))
				for b in ba:
					fh.write(bytearray([b]))

				ba = bytearray(struct.pack("f", numpy.float32(sample.imag)))
				for b in ba:
					fh.write(bytearray([b]))
			fh.close()
			clf()


freqs = [1000.0, 500.0, 0.0, -500.0, -1000.0]
gains = ['auto', 30, 40]

capture(891100000.0, 'gsm', freqs, gains)
capture(890725000.0, 'gsm', freqs, gains)
capture(868275000.0, 'gsm', freqs, gains)
capture(891700000.0, 'uplink', freqs, gains)
capture(103500000.0, 'fm', freqs, gains)
capture(91100000.0, 'fm', freqs, gains)
capture(101100000.0, 'fm', freqs, gains)
capture(300000000.0, 'noise', freqs, gains)
capture(251000000.0, 'noise', freqs, gains)
capture(400000000.0, 'noise', freqs, gains)
