import cv2
import numpy as np
import pyaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time


#FORMAT = pyaudio.paInt16 
FORMAT = pyaudio.paFloat32
CHANNELS = 1 #2
RATE = 22050
CHUNK = 1024
RECORD_SECONDS = 100
WAVE_OUTPUT_FILENAME = "file.wav"
frames_audio = []

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
                
stream2 = audio.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True)

#p = pyaudio.PyAudio()
#stream = p.open(format=pyaudio.paFloat32,
#                channels=1,
#                rate=rate,
#                input=True,
#                input_device_index=1,
#                frames_per_buffer=chunk_size)

frames = []

plt.figure(figsize=(10, 4))
do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db #Преобразуйте спектрограмму мощности (квадрат амплитуды) в единицы децибел (дБ)

while True:

    start = time.time()

    data = stream.read(CHUNK)
    stream2.write(data)
#    data = np.frombuffer(data, dtype=np.int16) 
    data = np.fromstring(data, dtype=np.float32)

    melspec = do_melspec(y=data, sr=RATE, n_mels=128, fmax=4000)
    norm_melspec = pwr_to_db(melspec, ref=np.max)
    print (melspec.shape, norm_melspec.shape)
    frames.append(norm_melspec)
    
    
    if len(frames) == 20:

        
        stack = np.hstack(frames)
        print (stack.shape)
        
        librosa.display.specshow(stack, y_axis='mel', fmax=4000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        #break
        frames.pop(0)
        
    


    t = time.time() - start

    #print(1 / t)
    
    
#https://gist.github.com/sshh12/62c740b329229c7292f2a7b520b0b6f3    
