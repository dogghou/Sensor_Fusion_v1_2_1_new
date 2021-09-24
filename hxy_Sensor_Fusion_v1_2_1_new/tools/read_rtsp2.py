# coding: utf-8
import datetime
import cv2
import os
import time
from threading import Thread
from pynput.keyboard import Key, Listener


def init_keyboard_listener():
    def on_press(key):
        global flag
        if key == Key.esc:
            flag = False

    def on_release(key):
        if key == Key.esc:
            return False

    def start():
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    T_listen = Thread(target=start, name='t1', args=())
    T_listen.start()


if __name__ == "__main__":
    init_keyboard_listener()
    ip = '192.168.6.169'.replace(".", "_")
    # rtsp = 'rtmp://192.168.48.111:11935/live/' + ip
    rtsp = "rtsp://admin:ehl1234.@10.130.210.30"
    # rtsp = "rtsp://admin:hik12345@10.0.0.13:554/h264/ch1/main/av_stream"

    # 初始化摄像头
    cap = cv2.VideoCapture(rtsp)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    size = (960, 540)
    flag = 1
    fps = 0.0
    frame_count = 0
    while flag:
        t1 = time.time()
        isSuccess, frame = cap.read()
        print(isSuccess)
        if isSuccess:

            # if frame_count % 120 == 0 or frame_count == 0:
            #     frame_count = 0
            #     i = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            #     filename = str(i) + '-' + ip + '.avi'
            #     print(filename)
                # video_writer = cv2.VideoWriter(filename, fourcc, 24, size)
            # video_writer.write(frame)
            # frame_count = frame_count + 1
            # print(frame_count)
            try:
                frame = cv2.resize(frame, size)
                print(time.time() - t1)
                # fps = (fps + (1. / (time.time() - t1))) / 2
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # if frame_count % 120 == 0:
                #     # video_writer.release()
                #     portion = os.path.splitext(filename)
                #     newname = portion[0] + '.mp4'
                #     # os.rename(filename, newname)
                cv2.imshow('show', frame)
            except: pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
