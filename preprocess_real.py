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



  
RATE = 22050
frames = []
vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
mel_list = []
wav_orig, _ = librosa.load("vcc2018/vcc2018_training/SF3/a0.wav", sr=RATE, mono=True)
spec = vocoder(torch.tensor([wav_orig]))

if spec.shape[-1] >= 64:    # training sample consists of 64 randomly cropped frames
    mel_list.append(spec.cpu().detach().numpy()[0])

mel_concatenated = np.concatenate(mel_list, axis=1)
mel_mean = np.mean(mel_concatenated, axis=1, keepdims=True)
mel_std = np.std(mel_concatenated, axis=1, keepdims=True) + 1e-9

mel_normalized = list()
for mel in mel_list:
    assert mel.shape[-1] >= 64, f"Mel spectogram length must be greater than 64 frames, but was {mel.shape[-1]}"
    app = (mel - mel_mean) / mel_std
    mel_normalized.append(app)

print (mel_normalized[0].shape)
print (wav_orig.shape, _, spec.shape, mel_mean.shape, mel_std.shape, mel_concatenated.shape)

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
        for i, sample in enumerate(tqdm(self.test_dataloader)):
                save_path = None
                real_B = sample
                print("real_B", i, real_B.shape)
                real_B = real_B.to(self.device, dtype=torch.float)
                fake_A = self.generator(real_B, torch.ones_like(real_B))
                #real_mel_A_fig = get_mel_spectrogram_fig(fake_A[0].detach().cpu())
                #imgs(real_mel_A_fig)
                
                wav_fake_A = decode_melspectrogram(self.vocoder, fake_A[0].detach(
                ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()

                wav_real_B = decode_melspectrogram(self.vocoder, real_B[0].detach(
                ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()

                save_path = os.path.join(self.converted_audio_dir, f"{i}-converted_{self.speaker_B_id}_to_{self.speaker_A_id}.wav")
                save_path_orig = os.path.join(self.converted_audio_dir,
                                         f"{i}-original_{self.speaker_B_id}_to_{self.speaker_A_id}.wav")
                wav_fake_A = wav_fake_A.cpu().detach().numpy()
                wav_real_B = wav_real_B.cpu().detach().numpy()
                writeS(save_path, self.sample_rate, wav_fake_A.T)


if __name__ == "__main__":
    parser = CycleGANTestArgParser()
    args = parser.parse_args()
    print (args)
    tester = MaskCycleGANVCTesting(args)
    tester.test()
#---------------------------------------->


#python3 -m mask_cyclegan_vc.test     --name mask_cyclegan_vc_VCC2SF3_VCC2TF1     --save_dir results/     --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training     --gpu_ids 0     --speaker_A_id VCC2SM5     --speaker_B_id SF3     --ckpt_dir results/mask_cyclegan_vc_VCC2SF3_VCC2TF1/ckpts     --load_epoch 800  --model_name generator_B2A


#python3 data_preprocessing/preprocess_vcc2018.py   --data_directory vcc2018/vcc2018_training   --preprocessed_data_directory vcc2018_preprocessed/vcc2018_training   --speaker_ids SF3 VCC2SM5



