from os import listdir, path
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
import torch

import numpy as np
import pyaudio
import librosa
import librosa.display

import matplotlib
import matplotlib.pyplot as plt
import time
import wave, struct
from moviepy.editor import * 
import io



def imgs(x):
    cv2.imshow('Rotat', np.array(x))
    cv2.waitKey(1)
    #cv2.destroyAllWindows()

def chunks(lst, count):
    start = 0
    for i in range(count):
          stop = start + len(lst[i::count, :])
          yield lst[start:stop, :]
          start = stop  

# moviepy
start_time = time.time()
clip = VideoFileClip("result_voice_vid_4.mp4")
n_frames = clip.reader.nframes
all_frames_video = []
all_frames_audio = []


plt.figure(figsize=(10, 4))
do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db #Преобразуйте спектрограмму мощности (квадрат амплитуды) в единицы децибел (дБ)

framerate = clip.audio.fps
# 44100 25.0
_s = clip.audio.to_soundarray(nbytes=4)
cut_wave = list(chunks(_s, int(clip.duration)))
print (_s.shape, clip.duration, _s.shape[0]/clip.duration, len(list(cut_wave)))


a_t = 0
g = 0

audio_frame = []
audio_time = []

for x in range(n_frames):
	frame = clip.get_frame(x)
	a_t +=  1

	if a_t == clip.fps:	
		signal = cut_wave[g]
		time = np.linspace(
			0, # start
			signal.shape[0] / framerate,
			num = signal.shape[0]
			)			
		
		plt.figure(1)
		plt.title("Sound Wave")
		plt.xlabel("Time")
		plt.plot(time, signal)
		plt.draw()
		plt.pause(0.00001)
		plt.clf()	
		g += 1
		a_t = 0
		print (g)

#------------------------------------------->

#https://medium.com/geekculture/real-time-audio-wave-visualization-in-python-b1c5b96e2d39

