# 1 Нужно захватывать видео
#
#
# 4 создавать мелспектограмму
# 5 генерировать голос
# 6 генерировать губы
# Повторять
# 

# 
import face_alignment
import cv2
import pyaudio
import wave
import numpy as np
from pathlib import PurePath, Path
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import time

###########################
# 3 записывать голос
###########################
FORMAT = pyaudio.paInt16
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
print ("recording...", int(RATE / CHUNK * RECORD_SECONDS))


  

###########################
# 2 Или использовать готовый файл в повторе
###########################

cap = cv2.VideoCapture('test.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)



while(cap.isOpened()):

    ret, frame = cap.read() 
    #cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    if ret:
        cv2.imshow("Image", frame)
        data = stream.read(CHUNK)
        array_a = np.frombuffer(data, dtype=np.int16)
        frames_audio.append(array_a)
        print (array_a.shape)
        
        time.sleep(fps/1000)
    else:
       print('no video')
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print ("stop Recording")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames_audio))
        waveFile.close()
        break
        
cap.release()
cv2.destroyAllWindows()  

  
