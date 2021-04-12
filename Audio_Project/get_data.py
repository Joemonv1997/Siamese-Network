import shutil
import librosa
import pandas as pd
import numpy as np
import sklearn
import glob
from random import randint
from sklearn.model_selection import train_test_split
from preprocessing import audio2Vector
dog = glob.glob("dog/*.wav")
sub_dog = glob.glob("Sub_dogs/*.wav")
cat = glob.glob("cat/*.wav")
def data_pair():
    pairs=[]
    labels=[]
    np.random.shuffle(sub_dog)
    np.random.shuffle(cat)
    for i in range(min(len(cat),len(sub_dog))):
        if (i%2)==0:
            pairs.append([audio2Vector(dog[randint(0,3)]),audio2Vector(cat[i])])
            labels.append(0)
        else:
            pairs.append([audio2Vector(dog[randint(0, 3)]), audio2Vector(sub_dog[i])])
            labels.append(1)
    return np.array(pairs,dtype="float32"),np.array(labels,dtype="float32")

X,y=data_pair()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
