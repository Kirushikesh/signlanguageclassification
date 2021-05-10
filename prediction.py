import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
import time

x,y=50,50
w,h=200,200
background = None
flag=0

prev='#'
count=0
skip=70
img = np.zeros((500, 800, 3), dtype="uint8")
cv2.putText(img,"PREDICTED TEXT - ", (70,100),cv2.FONT_HERSHEY_COMPLEX,1, (255,0,0), 1)

accumulated_weight = 0.5
cam = cv2.VideoCapture(0)
num_frames =0

def cal_accum_avg(frame, accumulated_weight):
    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

model = load_model("mycnn.h5")

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]

while(True):
    
    text=img.copy()

    ret,frame=cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy=frame.copy()
    cv2.rectangle(frame_copy, (x,y), (x+w,y+h), (255,128,0), 1)
    cv2.putText(frame_copy, "ROI",(125, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    roii= frame[y:y+h,x:x+w]
    gray_frame = cv2.cvtColor(roii, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 70:
        cal_accum_avg(gray_frame, accumulated_weight)        
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",(80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    else:
        diff = cv2.absdiff(background.astype("uint8"), gray_frame)
        _ , thresholded = cv2.threshold(diff,25, 255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            flag=0
            cv2.cv2.putText(frame_copy, "BACKGROUND",(80,400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        else:
            flag=1
            roi=frame_copy[y:y+h,x:x+w]
            roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            roi=cv2.resize(roi,(28,28))
            roi=roi/255
            roi=roi.reshape(28,28,1)
            pred=model.predict(np.expand_dims(roi,axis=0))
            pred=class_names[np.argmax(pred)]
            cv2.putText(text,pred,(450, 100),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

        if(not flag):
            if(prev!='#'):
                prev='#'
                count=0
        else:
            if(prev=='#'):
                prev=pred
                count=1
            elif(prev!='#'):
                if(prev==pred):
                    count+=1
                    if(count==30):
                        cv2.putText(img,pred, (skip,250),cv2.FONT_HERSHEY_TRIPLEX,1, (255,255,255), 1)
                        skip+=20
                        count=0
                else:
                    count=1
                    prev=pred
    num_frames+=1
    cv2.imshow("Sign Detection", frame_copy)
    cv2.imshow("Predictions",text)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()