# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:11:35 2019

@author: Radek
"""

import numpy as np
import scipy.signal as signal
import wave
import struct
import matplotlib.pyplot as plt

def allpass(input_signal, delay, gain):
    B = np.zeros(delay)
    B[0] = gain
    B[delay-1] = 1
    A = np.zeros(delay)
    A[0] = 1
    A[delay-1] = gain
    output_signal = np.zeros(input_signal.shape)
    output_signal = signal.lfilter(B, A, input_signal)
    return output_signal

def comb(input_signal, delay, gain):
    B = np.zeros(delay)
    B[delay-1] = 1
    A = np.zeros(delay)
    A[0] = 1
    A[delay-1] = -gain
    output_signal = np.zeros(input_signal.shape)
    output_signal = signal.lfilter(B, A, input_signal)
    return output_signal

def comb_with_lp(input_signal, delay, g, g1):
    g2 = g*(1-g1)
    B = np.zeros(delay+1)
    B[delay-1] = 1
    B[delay] = -g1
    A = np.zeros(delay)
    A[0] = 1
    A[1] = -g1
    A[delay-1] = -g2
    output_signal = np.zeros(input_signal.shape)
    output_signal = signal.lfilter(B, A, input_signal)
    return output_signal

def delay(input_signal, delay, gain = 1):
    output_signal = np.concatenate((np.zeros(delay), input_signal))
    output_signal = output_signal * gain
    return output_signal
    

def main():
#    OPENING / GENERATING TEST WAV SIGNAL   #

#    KRONECKER DELTA   #
    
#    sample = np.zeros((2,88200))
#    sample[:,0] = 1

#    WAV FILE   #
    
    sample_in = 'filename.wav'
    frame_rate = 44100.0

    wav_file = wave.open(sample_in, 'r')
    num_samples_sample = wav_file.getnframes()
    num_channels_sample = wav_file.getnchannels()
    sample = wav_file.readframes(num_samples_sample)
    total_samples_sample = num_samples_sample * num_channels_sample
    wav_file.close()
#    
    sample = struct.unpack('{n}h'.format(n = total_samples_sample), sample)
    sample = np.array([sample[0::2], sample[1::2]], dtype = np.float64)
    sample[0] /= np.max(np.abs(sample[0]), axis = 0)
    sample[1] /= np.max(np.abs(sample[1]), axis = 0)

#   INITIALIZATION OF ALGORITHM'S VARIABLES   #      
    
    stereospread  = 23
    delays_r = [2205, 2469, 2690, 2998, 3175, 3439]
    delays_l = [d + stereospread for d in delays_r]
    delays_early = [877, 1561, 1715, 1825, 3082, 3510]
    gains_early = [1.02, 0.818, 0.635, 0.719, 0.267, 0.242]
    g1_list = [0.41, 0.43, 0.45, 0.47, 0.48, 0.50]
    g = 0.9
    rev_to_er_delay = 1800
    allpass_delay = 286
    allpass_g = 0.7
    
    output_gain = 0.075
    dry = 1
    wet = 1
    width = 1
    wet1 = wet * (width / 2 + 0.5)
    wet2 = wet * ((1-width) / 2)
    
    early_reflections_r = np.zeros(sample[0].size)
    early_reflections_l = np.zeros(sample[1].size)
    combs_out_r = np.zeros(sample[0].size)
    combs_out_l = np.zeros(sample[1].size)
    
#   ALGORIITHM'S MAIN PART   #
   
    for i in range(6):
        early_reflections_r = early_reflections_r + delay(sample[0], delays_early[i], gains_early[i])[:sample[0].size]
        early_reflections_l = early_reflections_l + delay(sample[1], delays_early[i], gains_early[i])[:sample[1].size]
    
    for i in range(6):
        combs_out_r = combs_out_r + comb_with_lp(sample[0], delays_r[i], g, g1_list[i])
        combs_out_l = combs_out_l + comb_with_lp(sample[1], delays_l[i], g, g1_list[i])
    
    reverb_r = allpass(combs_out_r, allpass_delay, allpass_g)
    reverb_l = allpass(combs_out_l, allpass_delay, allpass_g)

    early_reflections_r = np.concatenate((early_reflections_r, np.zeros(rev_to_er_delay)))
    early_reflections_l = np.concatenate((early_reflections_l, np.zeros(rev_to_er_delay)))
    
    reverb_r = delay(reverb_r, rev_to_er_delay)
    reverb_l = delay(reverb_l, rev_to_er_delay)
    
    reverb_out_r = early_reflections_r + reverb_r
    reverb_out_l = early_reflections_l + reverb_l

    reverb_out_r = output_gain * ((reverb_out_r * wet1 + reverb_out_l * wet2) + np.concatenate((sample[0], np.zeros(rev_to_er_delay))) * dry)
    reverb_out_l = output_gain * ((reverb_out_l * wet1 + reverb_out_r * wet2) + np.concatenate((sample[1], np.zeros(rev_to_er_delay))) * dry)
  
#   WRITING TO FILE   #    
    
    signal_integer_r = (reverb_out_r*int(np.iinfo(np.int16).max)).astype(np.int16)
    signal_integer_l = (reverb_out_l*int(np.iinfo(np.int16).max)).astype(np.int16)
    
    signal_to_render = np.empty((signal_integer_r.size + signal_integer_l.size), dtype = np.int16)
    signal_to_render[0::2] = signal_integer_r
    signal_to_render[1::2] = signal_integer_l
    
    nframes = total_samples_sample
    comptype = "NONE"
    compname = "not compressed"
    nchannels = 2
    sampwidth = 2
    
    wav_file_write = wave.open('filename_out.wav', 'w')
    wav_file_write.setparams((nchannels, sampwidth, int(frame_rate), nframes, comptype, compname))

    for s in range(nframes):
        wav_file_write.writeframes(struct.pack('h', signal_to_render[s]))
        
    wav_file_write.close()
    
#   PLOTTING THE RESULTS   #
    
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(sample[0])
    plt.grid(True)
    plt.subplot(4,1,2)
    plt.plot(sample[1])
    plt.grid(True)
    plt.subplot(4,1,3)
    plt.plot(reverb_out_r)
    plt.grid(True)
    plt.subplot(4,1,4)
    plt.plot(reverb_out_l)
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()

