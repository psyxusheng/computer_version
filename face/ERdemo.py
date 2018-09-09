# -*- coding -*-
from face_tool import Face_cascade , Head_pos_estimator
from emotionPredictor import Predictor
import numpy as np
import cv2

FC = Face_cascade('resources')
HP = Head_pos_estimator('resources')
EP = Predictor()


cap = cv2.VideoCapture(0) # 从摄像头中取得视频

# 获取视频播放界面长宽
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

# 定义编码器 创建 VideoWriter 对象
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
#out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))


frames = []

while(cap.isOpened()):
    #读取帧摄像头
    ret, frame = cap.read()
    frames.append(np.expand_dims(frame,0))
    if ret == True:
        #输出当前帧
        #out.write(frame)
        face = FC.extract(frame)
        if type(face).__name__=='NoneType':
            cv2.putText(frame,'face not found',(width-400,height-100),cv2.FONT_HERSHEY_PLAIN,4.0,(0,0,255),2)
        elif face.shape[0]<64:
            continue
        else:
            pos = HP.getPos(face)
            emo,probs = EP.predict(face)
            cv2.putText(frame,emo,(width-400,height-100),cv2.FONT_HERSHEY_PLAIN,4.0,(0,0,255),2)
        cv2.imshow('My Camera',frame)
        #键盘按 Q 退出
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        if(cv2.waitKey(1) & 0xFF) == ord('r'):
            empty = np.zeros([1,height,width,3])
            frames.append(empty)
            
    else:
        break

# 释放资源
#out.release()
cap.release()
cv2.destroyAllWindows()

video = np.concatenate(frames,axis=0)
np.save('video',video)