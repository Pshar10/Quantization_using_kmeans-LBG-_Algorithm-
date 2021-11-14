from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import pyaudio
from sound import *
import pickle
import scipy.spatial.distance as sd
from task3 import *

voice = readAudio1b()
voice2 = readAudio1a()
voice2 = voice2[60000:120000]
   
#step one
Amplitude = np.max(voice) - np.min(voice) #Amplitude of training set
y = np.zeros((M,2))

trainSeq = np.reshape(voice,(-1,2))  #Initiate training set
Xk = np.reshape(voice2,(-1,2))  #Initiate Xk

f1 = pickle.load(open("codebook.bin","rb"))
f2 = pickle.load(open("voronoi_regions.bin","rb"))

plot2D(f1,f2,trainSeq)  
plot2D(f1,f2,Xk) 
