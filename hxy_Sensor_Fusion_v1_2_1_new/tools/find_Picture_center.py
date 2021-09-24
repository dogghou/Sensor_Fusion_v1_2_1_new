# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:37:57 2019

@author: Administrator
用一个小框，手动移动，寻找图中特定点，输出该点的图像坐标

"""

from pynput.keyboard import Key, Listener
from threading import Thread
import cv2,copy

global flag, y0, x0
flag = True

def on_press(key):
    global flag, y0, x0
    if key == Key.left:
        x0 = x0-1
    elif key == Key.right:
        x0 = x0+1
    elif key == Key.up:
        y0 = y0-1
    elif key == Key.down:
        y0 = y0+1
    if key == Key.esc:
        flag = False
def on_release(key):
    # print('{0} release'.format(key))
    if key == Key.esc:
        return False

def start():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
        
if __name__ == '__main__':
    T_listen = Thread(target=start, name='t1', args=())
    T_listen.start()
    
    point_name = 'B11'
    frame_path = 'D:/location6_south/12.5/Pic/' + point_name + '/000002.jpg'
    frame = cv2.imread(frame_path)
    # frame = cv2.resize(frame, (1280, 720))

    (y,x,h) = frame.shape
    print ((x,y));
    (y0,x0)= (int(y/2),int(x/2))
    
        
    while flag:
        img = copy.copy(frame)
        box = [x0-4,y0-4,x0+4,y0+4]
        img = cv2.rectangle(img, (box[0],box[1]),(box[2],box[3]),(255,255,255), 2)
        Text = str(x0)+', '+str(y0)
        img = cv2.putText(img, Text, (x-150,y-70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('Display', img)
        cv2.waitKey(10)        
        pass
    print (point_name)
    print ((x0, y0))
    print ((round(x0/float(x),5),round(y0/float(y),5)))
    cv2.destroyAllWindows()