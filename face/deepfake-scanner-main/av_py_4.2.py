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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
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
#clip = VideoFileClip("result_voice_vid_4.mp4")
clip = VideoFileClip("0001-0484.mp4")
n_frames = clip.reader.nframes
all_frames_video = []
all_frames_audio = []

# Matplot
fig = Figure(figsize=(10, 4), dpi=100)
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)

framerate = clip.audio.fps
_s = clip.audio.to_soundarray(nbytes=4)#[:,0]
size = int(_s.shape[0]/n_frames)

# 44100 25.0

#cut_wave = np.lib.stride_tricks.as_strided(_s, shape=(n_frames, size), 
#strides = _s.strides*2)


#def sliding_window(elements, window_size):
#    if len(elements) <= window_size:
#        return elements
#    for i in range(len(elements) - window_size + 1):
#        yield elements[i:i+window_size]
#cut_wave = sliding_window(_s, size)
#print(next(cut_wave).shape)



##cut_wave = list(chunks(_s, int(clip.duration)))
cut_wave = list(chunks(_s, int(n_frames)))
##print (_s.shape, clip.duration, _s.shape[0]/clip.duration, len(list(cut_wave)))


a_t = 0
g = 0

audio_frame = []
audio_time = []



new_data = []

##1 сек = 1000 млсек

##1/25 = 0.04
##1/44100 = 0.00002

print (len(list(cut_wave)), n_frames)
for x in range(n_frames):
	frame = clip.get_frame(x)
	a_t +=  1
	signal = cut_wave[x][:,0]
	audio_frame.append(signal)
	if a_t == 4:	
		signal = np.hstack(audio_frame)
		print (signal.shape)
		time_line = np.linspace(
			0, # start
			signal.shape[0] / framerate,
			num = signal.shape[0]
			)			
		plt.title("Sound Wave")
		plt.xlabel("Time")
		ax.plot(time_line, signal)
		canvas.draw()
		image = canvas.buffer_rgba()
		new_data.append(np.array(image))
		imgs(image)
		ax.clear()
		a_t = 0
		
		if signal.shape[0] >= framerate:
			audio_frame = audio_frame[framerate//2:]
		else:
			audio_frame.pop(0)
		print (g, image.shape)
		


				
clip = ImageSequenceClip(new_data, fps = 6)	
clip.ipython_display(width = 360) 	
#------------------------------------------->

#https://medium.com/geekculture/real-time-audio-wave-visualization-in-python-b1c5b96e2d39

