import cv2
import subprocess
def pipe_start():
    # rtsp = 'rtsp://admin:ehl1234.@{}:554/h264/ch1/main/av_stream'.format('10.130.210.30')
    # rtmp = 'rtmp://10.130.212.31:10009/origin'
    rtmp = 'rtmp://10.130.212.19:1935/live'

    # 读取视频并获取属性
    # cap = cv2.VideoCapture(rtsp)
    # Get video information
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ffmpeg command
    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', "{}x{}".format(960, 540),
               # '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-f', 'flv',
               rtmp]

    # 管道配置
    p = subprocess.Popen(command, stdin=subprocess.PIPE)
    return p

