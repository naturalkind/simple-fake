from __future__ import division, print_function, absolute_import
import numpy as np,wave
import scipy as sp
import matplotlib.pyplot as plt
import PIL.Image as Image
import os,sys
import librosa
import librosa.display
from pylab import *
import shutil
import time, cv2
from moviepy.editor import * 

# if you want to see utterance mel spectrogram and delta,delta-delta picture,
# set __DEBUG_ as True,and the pictures will be DEBUG directory

def read_wav(wav_path):
#    wavefile = wave.open(wav_path)
#    nchannels,sampwidth,framerate,nframes,comptype,compname=wavefile.getparams()
#    strdata = wavefile.readframes(nframes)
#    wavedata = np.fromstring(strdata, dtype=np.int16).astype('float32')# / (2 ** 15)

#    #print ('nchnnels:%d'%nchannels)
#    #print ('sampwidth:%d'%sampwidth)
#    #print ('framerate:%d'%framerate)
#    #print ('nframes:%d'%nframes)
#    wavefile.close()
#    return nchannels,sampwidth,framerate,nframes,wavedata

    # Video File
    clip = VideoFileClip(wav_path) 
    fps = clip.fps
    wavedata = clip.audio.to_soundarray(nbytes=4)#[:,0]
    framerate = clip.audio.fps
    
    # Audio File
#    clip = AudioFileClip(wav_path)
#    wavedata = clip.to_soundarray(nbytes=4)#[:,0]
#    framerate = clip.fps 
#    fps = 25  
    
    
    return framerate, wavedata[:,0], fps
    
    
def gen_utterance_melspec(wav_path):
    """
    Compute a mel-scaled spectrogram to a utterance wavefile
    :param wav_path: audio time-series file
    :return:
    """
    framerate, wavedata, fps = read_wav(wav_path)
    Sxx = librosa.feature.melspectrogram(y=wavedata, 
                                         sr=framerate, 
                                         n_fft=(int)(fps*framerate/1000), 
                                         hop_length=(int)((10)*framerate/1000), 
                                         n_mels=128, fmin=20, fmax=8000)
    return Sxx


def gen_segments_melspec(X, window_size, overlap_sz):
    """
    Create an overlapped version of X

    Parameters
    ----------
    X : ndarray, shape=(n_mels,n_samples)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    overlap_sz : int
        Step size between windows

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    
    window_step = (window_size-overlap_sz)
    append = np.zeros((window_size, (window_step - (X.shape[-1]-overlap_sz) % window_step)))
    X = np.hstack((X, append))
    print (window_size, overlap_sz, window_size-overlap_sz, X.shape)
    new_shape = ((X.shape[-1] - overlap_sz) // window_step,window_size,X.shape[0])
    new_strides = (window_step*8,X.strides[0],X.strides[-1])
    X_strided = np.lib.stride_tricks.as_strided(X, shape=new_shape, strides=new_strides)

    return X_strided

def normlize(x):
    return ((x-np.min(x))/(np.max(x)-np.min(x)))



def gen_dcnn_input(wav_path):
    utterance_melspec = gen_utterance_melspec(wav_path)
    segments_melspec = gen_segments_melspec(utterance_melspec, window_size=128, overlap_sz=30)
    

    for num in range(0, segments_melspec.shape[0]):
        print (num)
        static = librosa.power_to_db(segments_melspec[num], ref=np.max)
        delta = librosa.feature.delta(static, order=1)
        delta2 = librosa.feature.delta(static, order=2)

        static = normlize(static)*255
        delta = normlize(delta)*255
        delta2 = normlize(delta2)*255

        images = np.dstack((static,delta,delta2))
        print (images.shape)
        
        cv2.imshow("img", images/255.)
        cv2.waitKey(1) 
        time.sleep(0.025)


if __name__ == '__main__':

#    wav_file = "TASCAM_0009.wav"
    wav_file = "result_voice_vid_1.mp4"

    gen_dcnn_input(wav_file)
    
