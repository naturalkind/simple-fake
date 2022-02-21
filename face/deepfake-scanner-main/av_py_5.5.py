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
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import io
from IPython import display


def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 

def imgs(img, T):
	if T == "image":
		winname = "image"
#		cv2.namedWindow(winname)        # Create a named window
#		cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
		cv2.imshow(winname, np.array(img))
		cv2.setMouseCallback('image', click_event)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	elif T == "video":
		cv2.imshow('Rotat', np.array(img))
		cv2.waitKey(1)
		
		
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
#fig, ax = plt.subplots(figsize=(13, 4), dpi=100)

#ax = fig.add_axes([0, 0, 1, 1])
#ax = fig.add_subplot(111)

framerate = clip.audio.fps
_s = clip.audio.to_soundarray(nbytes=4)#[:,0]
size = int(_s.shape[0]/n_frames)

cut_wave = list(chunks(_s, int(clip.duration*5)))

## Одно изображение ---->
time_line = np.linspace(
	0, # start
	_s.shape[0] / framerate,
	num = _s.shape[0]
	)
	
#print (time_line.shape, cut_wave[0].shape)			
#plt.title("Sound Wave")
#plt.xlabel("Time seconds")
#ax.set_aspect('equal')
#plt.grid()
#plt.vlines(x=0, ymin=min(_s[:,0]), ymax=max(_s[:,0]), linestyles ="solid", colors ="k")



#plt.plot(time_line, _s[:,0])



##plt.xlabel("Time milliseconds")
##plt.plot((10**3)*time_line, _s[:,0] / _s.shape[0]) 
#ax.set_xticks(np.arange(min(time_line),max(time_line),1))
#plt.show()

a_t=0
step=0.1




#for x in range(n_frames):
#		#ax.plot(time_line, _s[:,0])
#		ax.vlines(x=a_t, ymin=min(_s[:,0]), ymax=max(_s[:,0]), linestyles ="solid", colors ="k")
#		fig.canvas.draw()
#		image = fig.canvas.buffer_rgba()
#		imgs(image, "video")
#		ax.clear()
#		a_t += step

duration = clip.duration # in sec
refreshPeriod = 100 # in ms

fig,ax = plt.subplots(figsize=(10, 4), dpi=100)
vl = ax.axvline(0, ls='-', color='r', lw=1, zorder=10)
ax.set_xlim(0,duration)
ax.set_xticks(np.arange(min(time_line),max(time_line),1))
plt.plot(time_line, _s[:,0])

def animate(i,vl,period):
    t = i*period / 1000
    vl.set_xdata([t,t])
    fig.canvas.draw()
    image = fig.canvas.buffer_rgba()
    imgs(image, "video")
    return vl,

ani = animation.FuncAnimation(fig, animate, frames=int(duration/(refreshPeriod/1000)), fargs=(vl,refreshPeriod), interval=refreshPeriod)
#plt.show()
ani.save("movie.mp4")
### ---------------->

#https://hackernoon.com/audio-handling-basics-how-to-process-audio-files-using-python-cli-jo283u3y
#https://www.analyticsvidhya.com/blog/2019/07/learn-build-first-speech-to-text-model-python/	
#https://learn.sparkfun.com/tutorials/graph-sensor-data-with-python-and-matplotlib/update-a-graph-in-real-time
#https://stackoverflow.com/questions/61808191/is-there-an-easy-way-to-animate-a-scrolling-vertical-line-in-matplotlib

