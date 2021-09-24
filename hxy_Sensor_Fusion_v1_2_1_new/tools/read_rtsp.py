# coding: utf-8
import datetime
import cv2
import os
import time
from threading import Thread
from pynput.keyboard import Key, Listener
import ffmpeg
import skimage
import imageio


imageio.plugins.ffmpeg.download()

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
    # 子进程
    from pathlib import Path
    import ffmpeg


    (
        ffmpeg
            # .input('rtsp://{}@{}:554/h264/ch1/main/av_stream'.format("admin:ehl1234.", host))
            .input("rtsp://admin:ehl1234.@10.130.210.59:554/h264/ch1/main/av_stream")
            # 保存的文件名
            .output('saved_rtsp.mp4')
            # 覆盖同名文件
            .overwrite_output()
            # 运行保存
            .run(capture_stdout=True)
    )

    # (
    #     ffmpeg
    #     .input("rtsp://admin:ehl1234.@10.130.210.59:554/h264/ch1/main/av_stream")
    #     .output('save_rtsp.mp4', format='h264')
    #     .run_async(pipe_stdout=True)
    # )
