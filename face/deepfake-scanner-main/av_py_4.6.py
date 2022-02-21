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
#clip = VideoFileClip("0001-0484.mp4") 
clip = VideoFileClip("result_voice_vid_4.mp4") 

n_frames = clip.reader.nframes
all_frames_video = []
all_frames_audio = []

# Matplot
#fig = Figure(figsize=(10, 4), dpi=100)
#canvas = FigureCanvas(fig)
fig, ax = plt.subplots()

#ax = fig.add_axes([0, 0, 1, 1])
#ax = fig.add_subplot(111)

framerate = clip.audio.fps
_s = clip.audio.to_soundarray(nbytes=4)#[:,0]
size = int(_s.shape[0]/n_frames)

cut_wave = list(chunks(_s, int(clip.duration*5)))

## Одно изображение ---->
#time_line = np.linspace(
#	0, # start
#	_s.shape[0] / framerate,
#	num = _s.shape[0]
#	)
#	
#print (time_line.shape, cut_wave[0].shape)			
#plt.title("Sound Wave")
#plt.xlabel("Time milliseconds")
##ax.set_aspect('equal')
##plt.plot(time_line, _s[:,0])
#plt.plot((10**3)*time_line, _s[:,0] / _s.shape[0]) 
#plt.show()
## ---------------->

do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db #Преобразуйте спектрограмму мощности (квадрат амплитуды) в единицы децибел (дБ)

a_t = 0
g = 0
new_data = []

print (len(list(cut_wave)), n_frames//len(cut_wave), n_frames, clip.fps)
for x in range(n_frames):
	frame = clip.get_frame(x)
	
	if a_t == n_frames//len(cut_wave):
		signal = cut_wave[g][:,0]
		melspec = do_melspec(y=signal, sr=framerate, n_mels=128, fmax=4000)
		norm_melspec = pwr_to_db(melspec, ref=np.max)
		
		
		time_line = np.linspace(
			0, # start
			signal.shape[0] / framerate,
			num = signal.shape[0]
			)			
		librosa.display.specshow(norm_melspec, y_axis='mel', fmax=4000, x_axis='time')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Mel spectrogram audio')
		#plt.draw()
		#canvas.draw()
		fig.canvas.draw()
		#plt.show()
		image = fig.canvas.buffer_rgba()
		new_data.append(np.array(image))
		imgs(image)
		#ax.clear()
		#fig.canvas.clear()
		plt.clf()
		
		print (signal.shape, g, x)
		a_t = 0
		g+=1		
	a_t +=  1
	
clip = ImageSequenceClip(new_data, fps = n_frames//len(cut_wave))	
clip.ipython_display(width = 360) 	
#------------------------------------------->
#https://stackoverflow.com/questions/70294656/plot-fourier-in-frequency-domain-of-voice-in-python
#https://medium.com/geekculture/real-time-audio-wave-visualization-in-python-b1c5b96e2d39

