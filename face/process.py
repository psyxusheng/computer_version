# -*- coding: utf-8 -*-
import os,re
import cv2

from face_tool import FaceDetector,Shape_calculator

faceDet = FaceDetector()
shapeDet = Shape_calculator()

def convertLabel(value):
    if value in [90,75,60]:
        return 4
    if value in [45,30,15]:
        return 3
    if value in [0]:
        return 2
    if value in [-15,-30,-45]:
        return 3
    if value in [-60,-75,-90]:
        return 4

def preprocess(folder):
    """
    90~60
    60~15
    0 font  
    -15~-60 
    -60 ~-90    
    """
    data = []
    for root,folder,fns in os.walk(folder):
        for fn in fns:
            if not fn.endswith('.jpg'):
                continue
            vert,heri = re.findall("([+\-]\d+)([+\-]\d+)\.jpg",fn)[0]
            v,h = int(vert),int(heri)
            label_v = convertLabel(v)
            label_h = convertLabel(h)
            picture = cv2.imread(os.path.join(root,fn))
            gray    = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
            #get face and 
            face_arr,face_rect = faceDet.getFace_arr(gray)
            if type(face_arr).__name__!='NoneType':
                landmarks = shapeDet.calculate(gray,face_rect)
                data.append([landmarks,label_v,label_h])
    return data



data = preprocess('headpose')

import numpy as np
x = np.array([i[0] for i in data],dtype=np.float32)
y = np.array([i[1:] for i in data],dtype=np.int32)

np.save('faceshape',x)
np.save('faceangel',y)

