import cv2
from picamera2 import Picamera2
import numpy as np
from tensorflow import keras
from time import sleep
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(18,GPIO.OUT)
GPIO.setup(15,GPIO.OUT)
p15=GPIO.PWM(15,50)
p15.start(0)
p18=GPIO.PWM(18,50)
p18.start(0)

time = 0
thresh=50
max_diff=10
a,b,c=None,None,None
isChecking = False

picam2=Picamera2()
model=keras.models.load_model("/home/sein/BottleModel.h5")
picam2.start(show_preview=False)

a = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
b = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
draw = None

p15.ChangeDutyCycle(6.5)
sleep(0.5)
GPIO.setup(15,GPIO.IN)

p18.ChangeDutyCycle(6)
sleep(0.5)
GPIO.setup(18,GPIO.IN)

sleep(1)

while True:
    time -= 500
    if not isChecking:
        c = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
        draw = c.copy()
    
        a_gray=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        b_gray=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
        c_gray=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY)
    
        diff1=cv2.absdiff(a_gray,b_gray)
        diff2=cv2.absdiff(b_gray,c_gray)
    
        ret,diff1_t=cv2.threshold(diff1,thresh,255,cv2.THRESH_BINARY)
        ret,diff2_t=cv2.threshold(diff2,thresh,255,cv2.THRESH_BINARY)
    
        diff=cv2.bitwise_and(diff1_t,diff2_t)
    
        k=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        diff=cv2.morphologyEx(diff,cv2.MORPH_OPEN,k)
        diff_cnt = cv2.countNonZero(diff)
        if diff_cnt > max_diff and not time > 0:
            isChecking = True
            time = 2000
        else:
            a = b
            b = c
    else:
        if time <= 0:
            time = 5000
            isChecking = False
            draw = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
            pred_img = draw / 255.0
            pred_img = cv2.resize(pred_img, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
            pred_test = pred_img.reshape(-1, 300, 300, 3)
            pred = model.predict(pred_test)
            result = np.where(pred[0] == max(pred[0]))[0][0]
            if(result == 2):
                GPIO.setup(18,GPIO.OUT)
                p18.ChangeDutyCycle(2.5)
                sleep(0.5)
                GPIO.setup(18,GPIO.IN)
                
                GPIO.setup(18,GPIO.OUT)
                p18.ChangeDutyCycle(6)
                sleep(0.5)
                GPIO.setup(18,GPIO.IN)
            else:
                GPIO.setup(18,GPIO.OUT)
                p18.ChangeDutyCycle(9.5)
                    sleep(0.5)
                GPIO.setup(18,GPIO.IN)
                
                if(result == 0):
                    GPIO.setup(15,GPIO.OUT)
                    p15.ChangeDutyCycle(12)
                    sleep(0.5)
                    GPIO.setup(15,GPIO.IN)
                else:
                    GPIO.setup(15,GPIO.OUT)
                    p15.ChangeDutyCycle(0.5)
                    sleep(0.5)
                    GPIO.setup(15,GPIO.IN)
                    
                GPIO.setup(15,GPIO.OUT)
                GPIO.setup(18,GPIO.OUT)
                p15.ChangeDutyCycle(6.5)
                p18.ChangeDutyCycle(6)
                sleep(0.5)
                GPIO.setup(15,GPIO.IN)
                GPIO.setup(18,GPIO.IN)
    if cv2.waitKey(500) & 0xFF == 27:
        break
p15.stop()
p18.stop()
GPIO.cleanup()
