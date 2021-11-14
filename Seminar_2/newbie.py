from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from sound import *
import pickle
import scipy.spatial.distance as sd
from scipy.spatial import voronoi_plot_2d,Voronoi
from play_clip import mod as m
   
samplerate, speech = wav.read("speech.wav")  # one channel or mono
speech = m.clipaudio(speech,samplerate,1,3)
data = speech

Codebook_Vector = np.zeros((256,2))

Amplitude =  np.max(data)-np.min(data)

M = 256

# print(np.max(data))
# print(np.min(data))
# print(np.max(data)-np.min(data))

Codebook_Vector = np.zeros((M,2))    #Initiate Codebook_Vector
for i in range(M):              
    Codebook_Vector[i,:] = np.array([Amplitude/M*i,Amplitude/M*i]) 

print(Codebook_Vector.shape)


Seq = np.linspace(np.min(data),np.max(data),M)

print(Seq.shape)
Codebook_Vector[:,0] = Seq    #Assigned Codebook_Vector vectors
Codebook_Vector[:,1] = Seq


print(Codebook_Vector)
