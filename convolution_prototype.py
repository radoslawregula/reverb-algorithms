# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:11:40 2019

@author: Radosław
"""

import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
from scipy.signal import convolve

def main():
    
#   OPENING TEST WAV FILE   #
    
    sample_in = 'filename_sample.wav'
    reverb_in = 'filename_reverb_impulse.wav'
    frame_rate = 44100.0

    wav_file = wave.open(sample_in, 'r')
    num_samples_sample = wav_file.getnframes()
    num_channels_sample = wav_file.getnchannels()
    sample = wav_file.readframes(num_samples_sample)
    total_samples_sample = num_samples_sample * num_channels_sample
    wav_file.close()
    
    wav_file = wave.open(reverb_in, 'r')
    num_samples_reverb = wav_file.getnframes()
    num_channels_reverb = wav_file.getnchannels()
    reverb = wav_file.readframes(num_samples_reverb)
    total_samples_reverb = num_samples_reverb * num_channels_reverb
    wav_file.close()

    sample = struct.unpack('{n}h'.format(n = total_samples_sample), sample)
    sample = np.array([sample[0::2], sample[1::2]], dtype = np.float64)
    sample[0] /= np.max(np.abs(sample[0]), axis = 0)
    sample[1] /= np.max(np.abs(sample[1]), axis = 0)
    
    reverb = struct.unpack('{n}h'.format(n = total_samples_reverb), reverb)
    reverb = np.array([reverb[0::2], reverb[1::2]], dtype = np.float64)
    reverb[0] /= np.max(np.abs(reverb[0]), axis = 0)
    reverb[1] /= np.max(np.abs(reverb[1]), axis = 0)
    
#   MAIN PART OF THE ALGORITHM   #
        
    gain_dry = 1
    gain_wet = 1
    output_gain = 0.05
    
    reverb_out = np.zeros([2, np.shape(sample)[1] + np.shape(reverb)[1] - 1], dtype = np.float64)
    reverb_out[0] = output_gain * (convolve(sample[0] * gain_dry, reverb[0] * gain_wet, method = 'fft'))
    reverb_out[1] = output_gain * (convolve(sample[1] * gain_dry, reverb[1] * gain_wet, method = 'fft'))

#   WRITING TO FILE   # 
    
    reverb_integer = np.zeros((reverb_out.shape))
    
    reverb_integer[0] = (reverb_out[0]*int(np.iinfo(np.int16).max)).astype(np.int16)
    reverb_integer[1] = (reverb_out[1]*int(np.iinfo(np.int16).max)).astype(np.int16)
    
    reverb_to_render = np.empty((reverb_integer[0].size + reverb_integer[1].size), dtype = np.int16)
    reverb_to_render[0::2] = reverb_integer[0]
    reverb_to_render[1::2] = reverb_integer[1]
   
    nframes = total_samples_sample
    comptype = "NONE"
    compname = "not compressed"
    nchannels = 2
    sampwidth = 2
    
    wav_file_write = wave.open('filename_out.wav', 'w')
    wav_file_write.setparams((nchannels, sampwidth, int(frame_rate), nframes, comptype, compname))

    for s in range(nframes):
        wav_file_write.writeframes(struct.pack('h', reverb_to_render[s]))
        
    wav_file_write.close()
    
#   PLOTTING THE RESULTS   #
  
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(sample[0])
    plt.xlim(0, num_samples_sample)
    plt.grid(True)
    plt.subplot(4,1,2)
    plt.plot(sample[1])
    plt.xlim(0, num_samples_sample)
    plt.grid(True)
    plt.subplot(4,1,3)
    plt.plot(reverb_out[0])
    plt.xlim(0, num_samples_sample)
    plt.grid(True)
    plt.subplot(4,1,4)
    plt.plot(reverb_out[1])
    plt.xlim(0, num_samples_sample)
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()
  
