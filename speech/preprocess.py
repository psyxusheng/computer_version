# -*- coding: utf-8 -*-
import os,math
import numpy  as np
import librosa
import matplotlib.pyplot as plt
from spectrumGraph import SpectrumFrames

"""
should including data augmentation
including: clip , roll , tune , add noise
    clip 1s(1000ms) 
    roll ?
"""
def do_roll(inputs,sr):
    nframes = math.ceil(inputs.shape[0]/sr)
    rolled = [inputs]
    for i in range(nframes):
        rolled.append(np.roll(inputs,sr*(i+1)))
    return rolled

def add_random_noise(inputs,scale=0.01):
    wn = np.random.randn(inputs.shape[0])
    y = np.where(np.abs(inputs) > 0.01, inputs + 0.01 * wn, inputs)
    return y



root = 'C:/Users/G7/Desktop/CASIA database'

def main(max_length_ms = 3000,sample_rate=8000):
    max_length_frame = max_length_ms//1000*sample_rate
    specs ,labels= [],[]
    mx,my=0,0
    minval = 0
    label_idx = -1
    for fd in os.listdir(root):
        label_idx+=1
        fp  = os.path.join(root,fd)
        for fn in os.listdir(fp):
            fn = os.path.join(fp,fn)
            try:
                frames_,rate = librosa.load(fn,sr=sample_rate)
                frames = frames_[:max_length_frame]
                for i in range(5):
                    noised = add_random_noise(frames,0.02)
                    rolled = do_roll(noised,rate)
                    for roll in rolled:
                        spec = SpectrumFrames(roll,rate)
                        x,y = spec.shape
                        if x>mx:
                            mx = x
                        if y>my:
                            my = y
                        if spec.min()<minval:
                            minval = spec.min()
                        specs.append(spec)
                        labels.append(label_idx)
            except:
                print(fn)
            continue
    num = len(specs)
    ret = np.ones((num,mx,my))*-30
    for i in range(num):
        w,h = specs[i].shape
        ret[i,:w,:h] = specs[i] 
    return ret,np.array(labels,dtype=np.int32)

ret , labels =  main()
np.save('graphs',ret)
np.save('labels',labels)