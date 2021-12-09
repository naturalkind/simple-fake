# -*- coding: utf-8 -*-
import numpy as np
import pyaudio
import time
import librosa
import os
import argparse
import pickle
import glob
import random
import numpy as np
from tqdm import tqdm

import librosa
from librosa.filters import mel as librosa_mel_fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

import sounddevice as sd
from scipy.io.wavfile import write as writeS

import torch
import torch.utils.data as data
import torchaudio

from mask_cyclegan_vc.model import Generator, Discriminator
from args.cycleGAN_test_arg_parser import CycleGANTestArgParser
from dataset.vc_dataset import VCDataset
from mask_cyclegan_vc.utils import decode_melspectrogram, get_mel_spectrogram_fig
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver
import cv2
import wave

FORMAT = pyaudio.paFloat32#pyaudio.paInt16
CHANNELS = 1 #2
#RATE = 44100
RATE = 22050
CHUNK = 1024#22050 #1024
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "file.wav"
  
audio = pyaudio.PyAudio()
  
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print ("recording...", int(RATE / CHUNK * RECORD_SECONDS))
#frames = []
vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

#for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#    data = stream.read(CHUNK)
#    array_a = np.frombuffer(data, dtype=np.float32)
#    spec = vocoder(torch.tensor([array_a]))
#    #array_a = np.frombuffer(data, dtype=np.int16)
#    #array_a = np.stack((array_a[::2], array_a[1::2]), axis=0) 
#    frames.append(array_a)
#    print (i, array_a.shape, len(data), int(RATE / CHUNK * 1), spec.shape)
#    
#numpydata = np.hstack(frames) 
#spec = vocoder(torch.tensor([numpydata]))
#print (numpydata.shape, len(frames), spec.shape) 

class MaskCycleGANVCTesting(object):
    """Tester for MaskCycleGAN-VC
    """

    def __init__(self, args):
        """
        Args:
            args (Namespace): Program arguments from argparser
        """
        # Store Args
        self.device = args.device
        self.converted_audio_dir = os.path.join(args.save_dir, args.name, 'converted_audio')
        os.makedirs(self.converted_audio_dir, exist_ok=True)
        self.model_name = args.model_name

        self.speaker_A_id = args.speaker_A_id
        self.speaker_B_id = args.speaker_B_id
        # Initialize MelGAN-Vocoder used to decode Mel-spectrograms
        self.vocoder = torch.hub.load(
            'descriptinc/melgan-neurips', 'load_melgan')
        self.sample_rate = args.sample_rate

        # Initialize speakerA's dataset
        self.dataset_A = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, self.speaker_A_id, f"{self.speaker_A_id}_normalized.pickle"))
        dataset_A_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, self.speaker_A_id, f"{self.speaker_A_id}_norm_stat.npz"))
        self.dataset_A_mean = dataset_A_norm_stats['mean']
        self.dataset_A_std = dataset_A_norm_stats['std']
        print ("Initialize speakerA's dataset", self.dataset_A_mean.shape, self.dataset_A_std.shape)
        # Initialize speakerB's dataset
        self.dataset_B = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, self.speaker_B_id, f"{self.speaker_B_id}_normalized.pickle"))
        dataset_B_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, self.speaker_B_id, f"{self.speaker_B_id}_norm_stat.npz"))
        self.dataset_B_mean = dataset_B_norm_stats['mean']
        self.dataset_B_std = dataset_B_norm_stats['std']

        source_dataset = self.dataset_A if self.model_name == 'generator_A2B' else self.dataset_B
        self.dataset = VCDataset(datasetA=source_dataset,
                                 datasetB=None,
                                 valid=True)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           drop_last=False)

        # Generator
        self.generator = Generator().to(self.device)
        self.generator.eval()

        # Load Generator from ckpt
        self.saver = ModelSaver(args)
        self.saver.load_model(self.generator, self.model_name)

    def loadPickleFile(self, fileName):
        """Loads a Pickle file.

        Args:
            fileName (str): pickle file path

        Returns:
            file object: The loaded pickle file object
        """
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def test(self):
#            frames = []
#            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#                data = stream.read(CHUNK)
#                array_a = np.frombuffer(data, dtype=np.float32)
#                frames.append(array_a)
#            frames = np.hstack(frames)
#            print (np.array(frames).shape)
#            spec = vocoder(torch.tensor([frames]))
#            real_B = spec#sample
#            real_B = real_B.to(self.device, dtype=torch.float)
#            fake_A = self.generator(real_B, torch.ones_like(real_B))
#            wav_fake_A = decode_melspectrogram(self.vocoder, fake_A[0].detach(
#            ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
#            wav_fake_A = wav_fake_A.cpu().detach().numpy()
#                
#            save_path = os.path.join(self.converted_audio_dir, f"F-converted_{self.speaker_B_id}_to_{self.speaker_A_id}.wav")
#            writeS(save_path, self.sample_rate, wav_fake_A.T)    
#    
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                array_a = np.frombuffer(data, dtype=np.float32)
                spec = vocoder(torch.tensor([array_a]))
                real_B = spec#sample
                print("real_B", i, real_B.shape, self.dataset_A_mean.shape)
                real_B = real_B.to(self.device, dtype=torch.float)
                fake_A = self.generator(real_B, torch.ones_like(real_B))
                #real_mel_A_fig = get_mel_spectrogram_fig(fake_A[0].detach().cpu())
                #imgs(real_mel_A_fig)
                
               # mel_std = np.std(mel_concatenated, axis=1, keepdims=True) + 1e-9
                
                wav_fake_A = decode_melspectrogram(self.vocoder, fake_A[0].detach(
                ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
                wav_fake_A = wav_fake_A.cpu().detach().numpy()
                frames.append(wav_fake_A)
            #numpydata = np.hstack(np.array(frames)) 
            #print (np.array(frames).shape, numpydata.shape)
            save_path = os.path.join(self.converted_audio_dir, f"{i}-converted_{self.speaker_B_id}_to_{self.speaker_A_id}.wav")
            writeS(save_path, self.sample_rate, numpydata.T)


if __name__ == "__main__":
    parser = CycleGANTestArgParser()
    args = parser.parse_args()
    print (args)
    tester = MaskCycleGANVCTesting(args)
    tester.test()


#[0. 0. 0.] <class 'numpy.ndarray'> (66444,) 22050 torch.Size([1, 80, 259]) (80, 1) (80, 1) (80, 259) (80, 259)

#---------------------------------------->


#python3 -m mask_cyclegan_vc.test     --name mask_cyclegan_vc_VCC2SF3_VCC2TF1     --save_dir results/     --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training     --gpu_ids 0     --speaker_A_id VCC2SM5     --speaker_B_id SF3     --ckpt_dir results/mask_cyclegan_vc_VCC2SF3_VCC2TF1/ckpts     --load_epoch 800  --model_name generator_B2A


#python3 data_preprocessing/preprocess_vcc2018.py   --data_directory vcc2018/vcc2018_training   --preprocessed_data_directory vcc2018_preprocessed/vcc2018_training   --speaker_ids SF3 VCC2SM5



