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

# if you want to see utterance mel spectrogram and delta,delta-delta picture,
# set __DEBUG_ as True,and the pictures will be DEBUG directory

def read_wav(wav_path):
    wavefile = wave.open(wav_path)
    nchannels,sampwidth,framerate,nframes,comptype,compname=wavefile.getparams()
    strdata = wavefile.readframes(nframes)
    wavedata = np.fromstring(strdata, dtype=np.int16).astype('float32')# / (2 ** 15)

    #print ('nchnnels:%d'%nchannels)
    #print ('sampwidth:%d'%sampwidth)
    #print ('framerate:%d'%framerate)
    #print ('nframes:%d'%nframes)
    wavefile.close()
    return nchannels,sampwidth,framerate,nframes,wavedata

def gen_utterance_melspec(wav_path):
    """
    Compute a mel-scaled spectrogram to a utterance wavefile
    :param wav_path: audio time-series file
    :return:
    """
    nchannels,sampwidth,framerate,nframes,wavedata = read_wav(wav_path)
    #wavedata,framerate = librosa.core.load(wav_path)

    # method 1:hamming window
    # sp.signal.get_window('hamming', 7)
    # Zxx = librosa.core.stft(wavedata, n_fft=25*framerate/1000, hop_length=(25-10)*framerate/1000, window='hamming', center=True, pad_mode='reflect')
    # Sxx = librosa.feature.melspectrogram(S=np.abs(Zxx),n_mels=64, fmin=20, fmax=8000)
    # method 2:hanning window
    Sxx = librosa.feature.melspectrogram(y=wavedata,sr=framerate,n_fft=(int)(25*framerate/1000),hop_length=(int)((10)*framerate/1000),n_mels=128,fmin=20,fmax=8000)
    #librosa.feature.mfcc()
    #librosa.time_to_frames()
    #print(Sxx.shape)
    return Sxx

def save_utterance(X,savepath,filename="melSpec"):
    # Convert a power spectrogram (amplitude squared) to decibel (dB) units
    X = librosa.power_to_db(X, ref=np.max)
    # Display a spectrogram/chromagram/cqt/etc.
    #librosa.display.specshow(X,fmin=20, fmax=8000)
    #plt.savefig("%s/%s"%(savepath,filename),bbox_inches='tight',pad_inches=0)
    plt.imsave("%s/%s"%(savepath,filename),X)
    close()

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
    append = np.zeros((128,(window_step - (X.shape[-1]-overlap_sz) % window_step)))
    X = np.hstack((X, append))
    # append zeros of end of X to get integer numbers of n_windows
    #print(X.shape)
    new_shape = ((X.shape[-1] - overlap_sz) // window_step,window_size,X.shape[0])
    new_strides = (window_step*8,X.strides[0],X.strides[-1])
    X_strided = np.lib.stride_tricks.as_strided(X, shape=new_shape, strides=new_strides)

    return X_strided

def normlize(x):
    return ((x-np.min(x))/(np.max(x)-np.min(x)))

def save_segment(X,pic_path):
    # librosa.display.specshow(X, fmin=20, fmax=8000)
    # plt.savefig(pic_path, bbox_inches='tight', pad_inches=0)
    plt.imsave(pic_path,X)
    #sp.misc.imsave(pic_path, X)
    close()

def save_dcnn_input(X):
    #plt.imsave(pic_path,X)
    sp.misc.imsave("OX.jpg",X)
    #sp.misc.toimage(X).save(pic_path)
    close()
    # pri_image = Image.open(pic_path)
    # pri_image.resize((227,227),Image.ANTIALIAS).save(pic_path)


def gen_dcnn_input(wav_path):
    utterance_melspec = gen_utterance_melspec(wav_path)
    segments_melspec = gen_segments_melspec(utterance_melspec, window_size=128,overlap_sz=64-30)
    

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
#        librosa.display.specshow(images, y_axis='mel', fmax=8000, x_axis="time")
#        plt.colorbar(format='%+2.0f dB')
#        plt.title('Mel spectrogram')
#        plt.draw()
#        plt.pause(10)
#        plt.clf()
        #save_dcnn_input(images)
        # close path file 

def wav_to_pics(wav_path,savepath,nwavs):
    utterance_melspec = gen_utterance_melspec(wav_path)
    segments_melspec = gen_segments_melspec(utterance_melspec,window_size=64,overlap_sz=64-30)
    
    
    for num in range(0, segments_melspec.shape[0]):
        
        static = librosa.power_to_db(segments_melspec[num], ref=np.max)
        delta = librosa.feature.delta(static, order=1)
        delta2 = librosa.feature.delta(static, order=2)

        static = normlize(static)*255
        delta = normlize(delta)*255
        delta2 = normlize(delta2)*255

        images = np.dstack((static,delta,delta2))
                   
        save_dcnn_input(images)
        # close path file 

if __name__ == '__main__':

    wav_file = "main_voice.wav"

    gen_dcnn_input(wav_file)
    
