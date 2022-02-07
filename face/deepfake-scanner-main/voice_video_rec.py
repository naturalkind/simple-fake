from os import listdir, path
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
import torch
import platform
import time

import cv2
import numpy as np
import pyaudio
import librosa
import librosa.display

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import wave, struct

    

# length of data to read.
chunk = 1024


'''
************************************************************************
      This is the start of the "minimum needed to read a wave"
************************************************************************
'''
# open the file for reading.
wf = wave.open("main_voice.wav", 'rb')
nchannels, sampwidth, framerate, nframes, _, _ = wf.getparams()
print (nchannels, sampwidth, framerate, nframes, _, _, wf.getnframes() / wf.getframerate())
print(f'duration: {wf.getnframes() / wf.getframerate():.2f} seconds')
 
# create an audio object
p = pyaudio.PyAudio()

# open stream based on the wave object which has been input.
stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)

# read data (based on the chunk size)
data = wf.readframes(chunk)

frames = []

plt.figure(figsize=(10, 4))
do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db #Преобразуйте спектрограмму мощности (квадрат амплитуды) в единицы децибел (дБ)




from moviepy.video.io.bindings import mplfig_to_npimage
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io


out = cv2.VideoWriter(f'spectro_short_new.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (700, 280))

# play stream (looping from beginning of file to the end)
while data != '':
    try:
        # writing to the stream is what *actually* plays the sound.
        
        array_a = np.fromstring(data, dtype=np.int16).astype('float32')
        #print (array_a.shape)
        melspec = do_melspec(y=array_a, sr=framerate, n_mels=128, fmax=4000)
        norm_melspec = pwr_to_db(melspec, ref=np.max)
        #print (melspec.shape, norm_melspec.shape)
        frames.append(norm_melspec)
        if len(frames) == 20:

            stack = np.hstack(frames)
            
    ######### WORK ##################
#            librosa.display.specshow(stack, y_axis='mel', fmax=4000, x_axis='time')
#            plt.colorbar(format='%+2.0f dB')
#            plt.title('Mel spectrogram')
#            plt.draw()
#            
#            
#            buf = io.BytesIO()
#            plt.savefig(buf, format="png", dpi=70)
#            buf.seek(0)
#            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#            buf.close()
#            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
#            print (img.shape)
#            out.write(img)        
#            
#            
#            plt.pause(0.0001)
#            plt.clf()
    #########################
            #print (dir(plt))
            librosa.display.waveplot(stack, sr=wf.getframerate())#
            plt.title("Waveplot", fontdict=dict(size=18))
            plt.xlabel("Time", fontdict=dict(size=15))
            plt.ylabel("Amplitude", fontdict=dict(size=15))
            plt.draw()
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=70)
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            print (img.shape)
            out.write(img)
            
            plt.pause(0.0001)
            plt.clf()
    ##################

            
            frames.pop(0)



       # print (len(data), array_a.shape)
        #stream.write(data)
        data = wf.readframes(chunk)
    except KeyboardInterrupt:
        

        # cleanup stuff.
        stream.close()    
        p.terminate() 
        out.release()
    
#https://github.com/tzaiyang/SpeechEmoRec/blob/36a67c3fabc0fd92ff4c21509a1e3bc8ad025e94/melSpec.py
#https://gist.github.com/sshh12/62c740b329229c7292f2a7b520b0b6f3   
#https://dsp.stackexchange.com/questions/1593/improving-spectrogram-resolution-in-python
#https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056



 
