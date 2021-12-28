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
import pyaudio
import wave
import numpy as np
import scipy, cv2, os, sys, argparse, audio
from pathlib import PurePath, Path
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import time
from os import listdir, path

import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip

import platform
import librosa
import librosa.display



device = 'cuda' if torch.cuda.is_available() else 'cpu'
mel_step_size = 16

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

model = load_model("wav2lip.pth")
vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

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

Audio = pyaudio.PyAudio()
stream = Audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print ("recording...", int(RATE / CHUNK * RECORD_SECONDS))

stream2 = Audio.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True)
  

###########################
# 2 Или использовать готовый файл в повторе
###########################

cap = cv2.VideoCapture('test.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

plt.figure(figsize=(10, 4))
do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db


while(cap.isOpened()):

    ret, frame = cap.read() 
    #cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    if ret:
    
#        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
#        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

#        with torch.no_grad():
#         pred = model(mel_batch, img_batch)

#        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

#        for p, f, c in zip(pred, frames, coords):
#            y1, y2, x1, x2 = c
#            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

#            f[y1:y2, x1:x2] = p
#            out.write(f)
    
    
        # Video
        cv2.imshow("Image", frame)
        time.sleep(fps/1000)
        
        # Audio
        
        data = stream.read(CHUNK)
        array_a = np.frombuffer(data, dtype=np.int16)
        frames_audio.append(array_a)
        if len(frames_audio) == 16:
            mel = audio.melspectrogram(frames_audio)  
            frames_audio = []
            
            
        stream2.write(data) # Audio play online
        
        print (array_a, array_a.shape)
        
        
        
        
    else:
       print('no video')
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print ("stop Recording")
        stream.stop_stream()
        stream.close()
        Audio.terminate()
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(Audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames_audio))
        waveFile.close()
        break
        
cap.release()
cv2.destroyAllWindows()  

# FAQ

#https://gist.github.com/sshh12/62c740b329229c7292f2a7b520b0b6f3  

#https://medium.com/nuances-of-programming/%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7-%D0%B0%D1%83%D0%B4%D0%B8%D0%BE%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85-%D1%81-%D0%BF%D0%BE%D0%BC%D0%BE%D1%89%D1%8C%D1%8E-%D0%B3%D0%BB%D1%83%D0%B1%D0%BE%D0%BA%D0%BE%D0%B3%D0%BE-%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F-%D0%B8-python-%D1%87%D0%B0%D1%81%D1%82%D1%8C-1-2056fef8525e

#https://stackoverflow.com/questions/35970282/what-are-chunks-samples-and-frames-when-using-pyaudio  



#    def test(self):
#            frames = []
#            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#                data = stream.read(CHUNK)
#                array_a = np.frombuffer(data, dtype=np.float32)
#                frames.append(array_a)
#            frames = np.hstack(frames)
#            print (np.array(frames).shape, self.sample_rate)
#            spec = vocoder(torch.tensor([frames]))
#            real_B = spec#sample
#            real_B = real_B.to(self.device, dtype=torch.float)
#            fake_A = self.generator(real_B, torch.ones_like(real_B))
#            wav_fake_A = decode_melspectrogram(self.vocoder, fake_A[0].detach(
#            ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
#            wav_fake_A = wav_fake_A.cpu().detach().numpy()
#                
#            save_path = os.path.join(self.converted_audio_dir, f"F-converted_{self.speaker_B_id}_to_{self.speaker_A_id}.wav")
#            writeS(save_path, RATE, wav_fake_A.T)

#    def test(self):
#            frames = []
#            ix = 0
#            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#                data = stream.read(CHUNK)
#                array_a = np.frombuffer(data, dtype=np.float32)
#                frames.append(array_a)
#                if len(frames) == 30:
#                    frames = np.hstack(frames)
#                    print (np.array(frames).shape, self.sample_rate, self.converted_audio_dir)
#                    spec = vocoder(torch.tensor([frames]))
#                    real_B = spec#sample
#                    real_B = real_B.to(self.device, dtype=torch.float)
#                    fake_A = self.generator(real_B, torch.ones_like(real_B))
#                    wav_fake_A = decode_melspectrogram(self.vocoder, fake_A[0].detach(
#                    ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
#                    wav_fake_A = wav_fake_A.cpu().detach().numpy()
#                        
#                    #save_path = os.path.join(self.converted_audio_dir, f"F-converted_{self.speaker_B_id}_to_{self.speaker_A_id}.wav")
#                    save_path = os.path.join(self.converted_audio_dir, f"{ix}_NEW_TEST.wav")
#                    writeS(save_path, RATE, wav_fake_A.T)
#                    
#                    ix += 1
#                    frames = []
