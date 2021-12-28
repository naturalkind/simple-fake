# -*- coding: utf-8 -*-
import numpy as np
import pyaudio
import wave
import time
import librosa
import os
import argparse
import pickle
import glob
import random

from tqdm import tqdm

from librosa.filters import mel as librosa_mel_fn

import sounddevice as sd
from scipy.io.wavfile import write

# Простая визуализация
#def print_sound(indata, outdata, frames, time, status):
#    volume_norm = np.linalg.norm(indata)*10
#    print ("|" * int(volume_norm))

#with sd.Stream(callback=print_sound):
#    sd.sleep(10000)

# Запись голоса

FORMAT = pyaudio.paInt16
CHANNELS = 2
#RATE = 44100
RATE = 22050
CHUNK = 1024
RECORD_SECONDS = 100
WAVE_OUTPUT_FILENAME = "file.wav"
  
audio = pyaudio.PyAudio()
  
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print ("recording...", int(RATE / CHUNK * RECORD_SECONDS))
frames = []
IX = 0

try:
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        array_a = np.frombuffer(data, dtype=np.int16)
        frames.append(array_a)
#        frames.append(data)
    print ("finished recording")                                                        
except KeyboardInterrupt:
    #write('output2.wav', RATE, np.array(frames).astype('int16')) 
    print ("stop Recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

# Play audio realtime
#https://stackoverflow.com/questions/31674416/python-realtime-audio-streaming-with-pyaudio-or-something-else
