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

#FORMAT = pyaudio.paInt16
#CHANNELS = 1 #2
#RATE = 22050
#CHUNK = 1024
#RECORD_SECONDS = 100
#WAVE_OUTPUT_FILENAME = "file.wav"

#audio = pyaudio.PyAudio()
#  
## start Recording
#stream = audio.open(format=FORMAT, channels=CHANNELS,
#                rate=RATE, input=True,
#                frames_per_buffer=CHUNK)
#print ("recording...", int(RATE / CHUNK * RECORD_SECONDS))
#frames = []
#IX = 0

#try:
#    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#        data = stream.read(CHUNK)
#        array_a = np.frombuffer(data, dtype=np.int16)
#        frames.append(array_a)
#        print (array_a.shape, RATE / CHUNK * RECORD_SECONDS)
#    #print ("finished recording")                                                        
#except KeyboardInterrupt:
#    #write('output2.wav', RATE, np.array(frames).astype('int16')) 
#    print ("stop Recording")
#    stream.stop_stream()
#    stream.close()
#    audio.terminate()
#    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#    waveFile.setnchannels(CHANNELS)
#    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
#    waveFile.setframerate(RATE)
#    waveFile.writeframes(b''.join(frames))
#    waveFile.close()


  

###########################
# 2 Или использовать готовый файл в повторе
###########################1qa

# Create a VideoCapture object and read from input file
#cap = cv2.VideoCapture('test.mp4')
cap = cv2.VideoCapture('vm.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frames_array = []

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video  file")
   
# Read until video is completed
while(cap.isOpened()):
      
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame
    #cv2.imshow('Frame', frame)
    frames_array.append(np.array(frame))
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  # Break the loop
  else: 
    break
# When everything done, release 
# the video capture object
cap.release()
# Closes all the frames
#cv2.destroyAllWindows()

frames_array = np.array(frames_array)
print (frames_array.shape)

while True:
    for i in range(frames_array.shape[0]):
        cv2.imshow("Image", frames_array[i,:,:,:])
        cv2.waitKey(1)
        time.sleep(0.030)
        
###-------------------------------->

#while(cap.isOpened()):

#    ret, frame = cap.read() 
#    #cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
#    #cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#    if ret:
#        cv2.imshow("Image", frame)
#        time.sleep(fps/1000)
#    else:
#       print('no video')
#       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#cap.release()
#cv2.destroyAllWindows()    

#https://github.com/JRodrigoF/AVrecordeR/blob/master/AVrecordeR.py
#https://stackoverflow.com/questions/14140495/how-to-capture-a-video-and-audio-in-python-from-a-camera-or-webcam
#https://towardsdatascience.com/extracting-audio-from-video-using-python-58856a940fd
#https://coderedirect.com/questions/100223/audio-output-with-video-processing-with-opencv
