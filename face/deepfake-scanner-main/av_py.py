from os import listdir, path
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import time
import glob


import numpy as np
import pyaudio
import librosa
import librosa.display

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import wave, struct
from moviepy.editor import * 
import io


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
    new_shape = ((X.shape[-1] - overlap_sz) // window_step,window_size,X.shape[0])
    new_strides = (window_step*8,X.strides[0],X.strides[-1])
    X_strided = np.lib.stride_tricks.as_strided(X, shape=new_shape, strides=new_strides)

    return X_strided


def imgs(x):
    cv2.imshow('Rotat', np.array(x))
    cv2.waitKey(1)
    #cv2.destroyAllWindows()


# moviepy
start_time = time.time()
clip = VideoFileClip("result_voice_vid_4.mp4")
#n_frames = int(clip.fps * clip.duration)
n_frames = clip.reader.nframes
all_frames_video = []#list(clip.iter_frames())
all_frames_audio = []


plt.figure(figsize=(10, 4))
do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db #Преобразуйте спектрограмму мощности (квадрат амплитуды) в единицы децибел (дБ)

framerate = clip.audio.fps
print (framerate, clip.fps)
# 44100 25.0
#audio_ = to_soundarray()
_s = clip.audio.to_soundarray(nbytes=4)
#print (_s)
a_t = 0
slide_audio = 0
audio_frame = []
for x in range(n_frames):
	frame = clip.get_frame(x)
	imgs(frame)
	time.sleep(clip.fps/1000)
	all_frames_video.append(frame)
	all_frames_audio.append(clip.audio.get_frame(x))
	a_t +=  1
	if a_t == clip.fps:
		audio_cut = _s[slide_audio:slide_audio+framerate,0]
		print (audio_cut.shape)
		#(2048,)

		melspec = do_melspec(y=audio_cut, sr=framerate, n_mels=128, fmax=4000)
		norm_melspec = pwr_to_db(melspec, ref=np.max)
		audio_frame.append(norm_melspec)
		slide_audio += framerate
		a_t = 0
		
	if len(audio_frame) == 1:
		stack = np.hstack(audio_frame)	
		librosa.display.waveplot(stack, sr=framerate)#
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
		

		plt.pause(0.0001)
		plt.clf()
		
		audio_frame.pop(0)		
	
	#_s = clip.audio.to_soundarray(tt=x, fps=framerate, nbytes=2, buffersize=50000)
	#_s = clip.audio.to_soundarray()
	#a_time = time.time()
	#print (_s.shape, clip.duration, time.time()-a_time, framerate)
	#print (dir(clip.audio), clip.audio.get_frame(x).shape, frame.shape)
#	melspec = do_melspec(y=clip.audio.get_frame(x), sr=framerate, n_mels=128, fmax=4000)
#	norm_melspec = pwr_to_db(melspec, ref=np.max)
#	all_frames_audio.append(norm_melspec)
#	if len(all_frames_audio) == 20:
#		stack = np.hstack(all_frames_audio)	
#		librosa.display.waveplot(stack, sr=framerate)#
#		all_frames_audio.pop(0)
#		plt.title("Waveplot", fontdict=dict(size=18))
#		plt.xlabel("Time", fontdict=dict(size=15))
#		plt.ylabel("Amplitude", fontdict=dict(size=15))
#		plt.draw()

#		buf = io.BytesIO()
#		plt.savefig(buf, format="png", dpi=70)
#		buf.seek(0)
#		img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#		buf.close()
#		img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
#		print (img.shape)
#		

#		plt.pause(0.0001)
#		plt.clf()
#		
#		all_frames_audio.pop(0)
		
		
print (time.time()-start_time)
#wavedata = np.array(all_frames_audio).astype('float32')

#utterance_melspec = librosa.feature.melspectrogram(y=wavedata,sr=framerate,n_fft=(int)(25*framerate/1000),hop_length=(int)((10)*framerate/1000),n_mels=128,fmin=20,fmax=8000)
#segments_melspec = gen_segments_melspec(utterance_melspec, window_size=128,overlap_sz=64-30)



#print (dir(clip.audio), clip.audio.fps)


#	#print (x.shape)
##print (frames, n_frames)

#clip2 = ImageSequenceClip(all_frames)
#clip2.write_videofile("movie.mp4", fps=clip.fps)


#----------------------->
#start_time = time.time()
#info = reader.ffmpeg_parse_infos("result_voice_vid_4.mp4")
#clip = reader.FFMPEG_VideoReader("result_voice_vid_4.mp4")
#all_frames = []
##print (dir(reader), info["video_nframes"])
#for x in range(info["video_nframes"]):
#	frame = clip.get_frame(x)
#	all_frames.append(frame)
#print (time.time()-start_time)


