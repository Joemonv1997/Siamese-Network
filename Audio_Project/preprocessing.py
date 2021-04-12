import os
import glob
import shutil
import librosa
import pandas as pd
import numpy as np
os.chdir("D:/Siamese_Network/Audio_Project/Dataset/")

def audio2Vector(file):
    audio,sr=librosa.load(file,mono=True)
    audio=audio[::3]
    # To get audio embeddings
    mfcc=librosa.feature.mfcc(audio,sr=sr)
    pad_width=400-mfcc.shape[1]
    mfcc=np.pad(mfcc,pad_width=((0,0),(0,pad_width)),mode="constant")
    return mfcc
    
    
