# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:27:17 2019

@author: Radek
"""

import numpy as np
from scipy.linalg import circulant
import scipy.signal as signal
import wave
import struct
import matplotlib.pyplot as plt

def damping_filter_coeffs(delays, t_60, alpha):
    element_1 = np.log(10) / 4
    element_2 = 1 - (1 / (alpha ** 2))
    g = np.zeros(len(delays))
    p = np.zeros(len(delays))
    for i in range(len(delays)):
        g[i] = 10 ** ((-3 * delays[i] * (1/44100)) / t_60)
        p[i] = element_1 * element_2 * np.log10(g[i])
    print(g)
    print(p)
    return p, g

def delay(input_signal, delay, gain = 1):
    output_signal = np.concatenate((np.zeros(delay), input_signal))[:input_signal.size]
    output_signal = output_signal * gain
    return output_signal

def damping_filter(input_signal, p, g):
    B = np.array([g * (1 - p)])
    A = np.array([1, -p])
    output_signal = np.zeros(input_signal.shape)
    output_signal = signal.lfilter(B, A, input_signal)
    return output_signal

def tonal_correction_filter(input_signal, alpha):
    beta = (1 - alpha)/(1 + alpha)
    E_nomin = np.array([1, -beta])
    E_denomin = np.array([1-beta])
    output_signal = np.zeros(input_signal.shape)
    output_signal = signal.lfilter(E_nomin, E_denomin, input_signal)
    return output_signal
    
def main():
    
#   INITIALIZATION OF ALGORITHM'S VARIABLES   #      
 
    delay_lens = np.array([601, 1399, 1747, 2269, 2707, 3089, 3323, 3571, 3911, 4127, 4639, 4999])
    num_delay_lines = delay_lens.shape[0]
    b = 1
    c = 1
    gain_b = np.full((num_delay_lines, 1), b)
    gain_c = np.full((num_delay_lines, 2), c)
    gain_c[1::2, 0] *= -1
    gain_c[2::4, 1] *= -1
    gain_c[3::4, 1] *= -1
    init_delay = 0
    gain_dry = 1
    gain_wet = 1
    output_gain = 0.15
    alpha = 0.4
    t_60 = 1.5
    p_coeffs, g_coeffs = damping_filter_coeffs(delay_lens, t_60, alpha)
    fm_gain = 1

    permutation_matrix = circulant(np.concatenate((np.array([0,1]), np.zeros(len(delay_lens)-2))))
    N = permutation_matrix.shape[0]
    u_vector = np.ones((N,1))
    feedback_matrix = fm_gain * (permutation_matrix - np.matmul((2/N) * u_vector, u_vector.transpose()))
    
#    OPENING / GENERATING TEST WAV SIGNAL    #

#    KRONECKER DELTA    #
    
#    sample = np.zeros((2,88200))
#    sample[:,0] = 1
    
#    WAV FILE    #
    
    sample_in = 'filename.wav'
    frame_rate = 44100.0

    wav_file = wave.open(sample_in, 'r')
    num_samples_sample = wav_file.getnframes()
    num_channels_sample = wav_file.getnchannels()
    sample = wav_file.readframes(num_samples_sample)
    total_samples_sample = num_samples_sample * num_channels_sample
    wav_file.close()
    
    sample = struct.unpack('{n}h'.format(n = total_samples_sample), sample)
    sample = np.array([sample[0::2], sample[1::2]], dtype = np.float64)
    sample[0] /= np.max(np.abs(sample[0]), axis = 0)
    sample[1] /= np.max(np.abs(sample[1]), axis = 0)
    
    output_to_correct = np.zeros((sample.shape))
    output_wet = np.zeros((sample.shape))

#   MAIN LOOP    #
    
    for channel in range(2):
        
        print(np.shape(sample[channel].reshape(1,sample.shape[1])))

        sample_mx = np.tile(sample[channel,:], (num_delay_lines,1))
        sample_mx_out = np.zeros((sample_mx.shape))
        feedback_out_A = np.empty([num_delay_lines, sample_mx.shape[1]])
        feedback_out = np.zeros((feedback_out_A.shape))
        feedback_out_A = np.matmul(gain_b, sample[channel].reshape(1,sample.shape[1]))
        
        cnt = 0
    
        while True:
            for i in range(sample_mx.shape[0]):
                feedback_out_B = delay(feedback_out_A[i] + feedback_out[i], delay_lens[i])
                feedback_out_C = damping_filter(feedback_out_B, p_coeffs[i], g_coeffs[i]) 
                sample_mx_out[i] = feedback_out_C
                
            if np.array_equal(np.matmul(feedback_matrix, sample_mx_out), feedback_out):
                break
            cnt = cnt + 1
            feedback_out = np.matmul(feedback_matrix, sample_mx_out)

        print(cnt)
        
        output_to_correct[channel] = np.sum(sample_mx_out * gain_c[:, channel].reshape(gain_b.shape), axis = 0)
        output_wet[channel] = tonal_correction_filter(output_to_correct[channel], alpha)
        output_wet[channel] = delay(output_wet[channel], int(round(44.1 * init_delay)))
        
    output = output_gain * (output_wet * gain_wet + (sample * gain_dry)).reshape(sample.shape)
        
#    WRITING TO FINAL WAV FILE    #
    
    output_integer = np.zeros((output.shape))
    output_integer[0] = (output[0]*int(np.iinfo(np.int16).max)).astype(np.int16)
    output_integer[1] = (output[1]*int(np.iinfo(np.int16).max)).astype(np.int16)
    
    signal_to_render = np.empty((output_integer[0].size * 2), dtype = np.int16)
    signal_to_render[0::2] = output_integer[0]
    signal_to_render[1::2] = output_integer[1]
    
    nframes_reverb = total_samples_sample
    comptype = "NONE"
    compname = "not compressed"
    nchannels = 2
    sampwidth = 2
    
    wav_file_write = wave.open('filename_out.wav', 'w')
    wav_file_write.setparams((nchannels, sampwidth, int(frame_rate), nframes_reverb, comptype, compname))

    for s in range(nframes_reverb):
        wav_file_write.writeframes(struct.pack('h', signal_to_render[s]))
        
    wav_file_write.close()

#    PLOTTING THE RESULTS    #
    
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(np.arange(sample[0].shape[0])/44100, sample[0])
    plt.grid(True)
    plt.subplot(4,1,2)
    plt.plot(np.arange(sample[0].shape[0])/44100, sample[1])
    plt.grid(True)
    plt.subplot(4,1,3)
    plt.plot(np.arange(sample[0].shape[0])/44100, output[0])
    plt.grid(True)
    plt.subplot(4,1,4)
    plt.plot(np.arange(sample[0].shape[0])/44100, output[1])
    plt.grid(True)
    plt.show()    


if __name__ == "__main__":
    main()    
 