# -*- coding: utf-8 -*-

"""
import os
import numpy as np
import wave 
import matplotlib.pyplot as plt


 
f = wave.open(fd,'rb')
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)#读取音频，字符串格式
waveData = np.frombuffer(strData,dtype=np.int16)#将字符串转化为int
waveData = waveData#*1.0/(max(abs(waveData)))#wave幅值归一化
# plot the wave
time = np.arange(0,nframes)*(1.0 / framerate)
plt.plot(time,waveData)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Single channel wavedata")
plt.grid(1)#标尺，on：有，off:无。

"""
import os
import librosa
import numpy, wave
import matplotlib.pyplot as plt

# filename 是文件名
# window_length_ms 是以毫秒为单位的窗长
# window_shift_times 是帧移，是与窗长的比例 例如窗长20ms，帧移0.5就是10毫秒

def getSpectrum(filename, window_length_ms, window_shift_times):
    """
    window length : window_length_ms 
    window_shift-time :actually is rate : like windows length 20ms , shift:0.5 : then step is 10
    """
	# 读音频文件
    wav_file = wave.open(filename, 'rb')
    
    # 获取音频文件的各种参数
    params = wav_file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    
    # 获取音频文件内的数据，不知道为啥获取到的竟然是个字符串，还需要在numpy中转换成short类型的数据
    str_data = wav_file.readframes(wav_length)
    wave_data = numpy.frombuffer(str_data, dtype=numpy.short)
    
    # 将窗长从毫秒转换为点数
    window_length = int(framerate * window_length_ms / 1000)
    window_shift = int(window_length * window_shift_times)
    
    # 计算总帧数，并创建一个空矩阵
    nframe = int((wav_length - (window_length - window_shift)) / window_shift)
    
    spec = numpy.zeros((int(window_length/2), nframe))
    
    # 循环计算每一个窗内的fft值
    for i in range(nframe):
        start = i * window_shift
        end = start + window_length
        # [:window_length/2]是指只留下前一半的fft分量
        spec[:,i] = numpy.log(numpy.abs(numpy.fft.fft(wave_data[start:end])))[:window_length//2]
    return spec


def SpectrumWindows(filename, window_length_ms=20, step_size_ms=10,padding='same'):
    """
    split data in to windows with length and step_size 
    then calculate spectrum in each window also called frame
    return a series of spectrum graph
    """
	# read the audio file
    wav_file = wave.open(filename, 'rb')
    
    # read audio's parameters
    # they are 
    params = wav_file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    
    # read frames and transform to int8
    str_data = wav_file.readframes(wav_length)
    wave_data = numpy.frombuffer(str_data, dtype=numpy.short)
    
    # turn ms to num_frames
    window_length = framerate * window_length_ms // 1000
    step_length   = framerate * step_size_ms//1000
    
    # total frame size 
    #here the nframe is not the same frame in audio data , it's a window of  frames
    nframe = (wav_length -  window_length)//step_length + 1
    spec = numpy.zeros((int(window_length//2), nframe))
    #spec = numpy.zeros((window_length,nframe))
    # doing fast-furiors- decomposition
    for i in range(nframe):
        start = i * step_length
        end = start + window_length
        # only keep the half of it ? why !!!![:window_length/2]
        spec[:,i] = numpy.log(numpy.abs(numpy.fft.fft(wave_data[start:end]))+1)[:window_length//2]
    return spec


def SpectrumFrames(frames ,framerate, window_length_ms=20, step_size_ms=20):
    """
    split data in to windows with length and step_size 
    then calculate spectrum in each window also called frame
    return a series of spectrum graph
    """
	# read the audio file
    wave_length = frames.shape[0]
    window_length = framerate * window_length_ms // 1000
    step_length   = framerate * step_size_ms//1000
    
    # total frame size 
    #here the nframe is not the same frame in audio data , it's a window of  frames
    nframe = (wave_length -  window_length)//step_length + 1
    spec = numpy.zeros((nframe,int(window_length//2)))
    #spec = numpy.zeros((window_length,nframe))
    # doing fast-furiors- decomposition
    for i in range(nframe):
        start = i * step_length
        end = start + window_length
        # only keep the half of it ? why !!!![:window_length/2]
        spec[i,:] = numpy.log(numpy.abs(numpy.fft.fft(frames[start:end])))[:window_length//2]
    return spec


"""
fd = 'C:/Users/G7/Desktop/CASIA database/liuchanhg/fear/210.wav'

frames , rate = librosa.load(fd,sr=8000)
spectrum = SpectrumFrames(frames,rate,20,10)
plt.imshow(spectrum)
#noised_speech_spectrum = speech_spectrum[:,:300] + noise_spectrum[:, :300]

#plt.subplot(311)
#plt.imshow(speech_spectrum)

#plt.subplot(312)
#plt.imshow(noise_spectrum)

#plt.subplot(313)
#plt.imshow(noised_speech_spectrum)

"""