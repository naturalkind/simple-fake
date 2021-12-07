"""
Defines the util functions associated with the cycleGAN VC pipeline.
"""

import io
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchaudio
from torchvision.transforms import ToTensor

import librosa
import librosa.display

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def decode_melspectrogram(vocoder, melspectrogram, mel_mean, mel_std):
    """Decoded a Mel-spectrogram to waveform using a vocoder.

    Args:
        vocoder (torch.nn.module): Vocoder used to decode Mel-spectrogram
        melspectrogram (torch.Tensor): Mel-spectrogram to be converted
        mel_mean ([type]): Mean of the Mel-spectrogram for denormalization
        mel_std ([type]): Standard Deviations of the Mel-spectrogram for denormalization

    Returns:
        torch.Tensor: decoded Mel-spectrogram
    """
    mel_std = torch.from_numpy(mel_std)
    mel_mean = torch.from_numpy(mel_mean)
    print ("decode_melspectrogram >>>>>>>>>>>>>", type(melspectrogram), type(mel_std), type(mel_mean))
    denorm_converted = melspectrogram * mel_std + mel_mean
    rev = vocoder.inverse(denorm_converted.unsqueeze(0))
    return rev


def get_mel_spectrogram_fig(spec, title="Mel-Spectrogram"):
    """Generates a figure of the Mel-spectrogram and converts it to a tensor.

    Args:
        spec (torch.Tensor): Mel-spectrogram
        title (str, optional): Figure name. Defaults to "Mel-Spectrogram".

    Returns:
        torch.Tensor: Figure as tensor
    """
    figure, ax = plt.subplots()
    canvas = FigureCanvas(figure)
    S_db = librosa.power_to_db(10**spec.numpy().squeeze(), ref=np.max)
    img = librosa.display.specshow(S_db, y_axis='log', x_axis='time')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
        
    image = Image.open(buf)
    image = ToTensor()(image)
    print ("get_mel_spectrogram_fig >>>>>>>>>>>>>", image.shape, type(image))
    plt.close(figure)
    return image

    
            
            
        
    
