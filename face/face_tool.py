# -*- coding: utf-8 -*-
import os
import numpy as np
import dlib
import cv2
from numpy import array,int32
from numpy import ndarray
import tensorflow as tf
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

class FaceDetector():
    def __init__(self):
        self._face_detector  = dlib.get_frontal_face_detector()
    def getFace(self,img_fn,resize=None):
        picture = cv2.imread(img_fn)
        pic_array = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
        rects = [(face,face.area()) for face in self._face_detector(pic_array, 0)]
        if len(rects) == 0:
            return None,None
        rects.sort(key=lambda x:x[1])
        face = rects[-1][0]
        start_x,end_x,start_y,end_y = face.left(), face.right(), face.top(), face.bottom()
        face_array = pic_array[start_y:end_y,start_x:end_x]
        if resize:
            try:
                res = cv2.resize(face_array,resize,interpolation=cv2.INTER_CUBIC)
                return res,face
            except:
                return face_array,face
        return face_array,face
    
    def getFace_arr(self,arr,resize=None):
        rects = [(face,face.area()) for face in self._face_detector(arr, 0)]
        if len(rects) == 0:
            return None,None
        rects.sort(key=lambda x:x[1])
        face = rects[-1][0]
        start_x,end_x,start_y,end_y = face.left(), face.right(), face.top(), face.bottom()
        #face_array = arr[face.top():face.top()+face.height(),face.left():face.left()+face.width()]
        face_array = arr[start_y:end_y,start_x:end_x]
        #interpolation=cv2.INTER_CUBIC
        if resize:
            try:
                res = cv2.resize(face_array,resize,interpolation=cv2.INTER_AREA)
                return res,face
            except:
                return face_array,face
        return face_array,face


class Face_cascade():
    def __init__(self,resFolder):
        self._face_cascade = cv2.CascadeClassifier(resFolder+'/xml/haarcascade_frontalface_default.xml')
    def extract(self,fn_or_arr):
        if isinstance(fn_or_arr,str):
            array = cv2.imread(fn_or_arr)
            if not os.path.exists(fn_or_arr):
                raise Exception('no such file')
            #array = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
        elif isinstance(fn_or_arr,ndarray):
            array = fn_or_arr
        else:
            raise Exception('wrong input format , should be str(fn) or np.array with')
        faces = self._face_cascade.detectMultiScale(
                               array,
                               scaleFactor = 1.2,
                               minNeighbors = 4,
                               minSize = (20,20),
                               flags = cv2.CASCADE_SCALE_IMAGE
                            )
        if isinstance(faces,tuple) :
            return None
        max_area = -1
        coordinates = []
        for i in range(faces.shape[0]):
            x,y,w,h = faces[i,:]
            if max_area< w*h:
                coordinates = [x,y,w,h]
        x,y,w,h = coordinates
        return array[y:y+h,x:x+w]
    def convert(self,array,size=(48,48)):
        gray = cv2.cvtColor(array,cv2.COLOR_BGR2GRAY)
        res = cv2.resize(gray,size,interpolation=cv2.INTER_AREA)
        ret_ = np.expand_dims(res,-1)
        ret  = np.expand_dims(ret_,0)
        return ret
                    


class Shape_calculator():
    """
    下巴：8 
    鼻尖：30 
    左眼角：36 
    右眼角：45 
    左嘴角：48 
    右嘴角：54
    """
    def __init__(self,resFolder):
        self.postions = [30,36,45,8,60,54]
        self.postions_names = ['nose','left_eye','right_eye',
                               'lower_most face','left_lip','right_lip']
        self._shape_dectecor = dlib.shape_predictor(resFolder+'/shape_predictor_68_face_landmarks.dat')
    def calculate(self,face_array,face_obj):
        """
        with input shape : [a,b] two dimension
        """
        shapes = self._shape_dectecor(face_array,face_obj).parts()
        landmarks = [shapes[s] for s in self.postions]
        return array([[lm.x,lm.y] for lm in landmarks],dtype=int32)
    
    
class Head_pos_estimator():
    def __init__(self,resFolder):
        self.sess = tf.Session() #Launch the graph in a session.
        HPestimator = CnnHeadPoseEstimator(self.sess) #Head pose estimation object
        # Load the weights from the configuration folders
        HPestimator.load_yaw_variables(resFolder+"/head_pose/yaw/cnn_cccdd_30k.tf")
        HPestimator.load_roll_variables(resFolder+"/head_pose/roll/cnn_cccdd_30k.tf")
        HPestimator.load_pitch_variables(resFolder+"/head_pose/pitch/cnn_cccdd_30k.tf")
        self._estimator = HPestimator
    def getPos(self,face,with_label=True):
        """
        3-channals
        w=h >=64 array
        pith : nod or not up-down
        yaw: swing , or say , left-right
        roll : slope
        """
        roll = self._estimator.return_roll(face)
        pith = self._estimator.return_pitch(face)
        yaw = self._estimator.return_yaw(face)
        p,y,r = pith.flatten()[0],yaw.flatten()[0],roll.flatten()[0]
        if with_label:
            return ('up-down',p),('left-right',y),('slope',r)
        else:
            return p,y,r
        