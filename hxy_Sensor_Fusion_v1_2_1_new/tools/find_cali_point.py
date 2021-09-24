import cv2
import numpy as np
import pickle

WIN_NAME = 'pick_point'
txt_list = []

def onmouse_pick_point(event, x, y, flags, param):
    i = 1
    if event == cv2.EVENT_LBUTTONDOWN:
        print(i, x, y)
        txt_list.append((i, x, y))
        i += 1
        cv2.drawMarker(param, (x, y), (0, 0, 255))

if __name__ == '__main__':
    image = cv2.imread("E:/Desketop/1.jpg")
    cv2.namedWindow(WIN_NAME)
    cv2.setMouseCallback(WIN_NAME, onmouse_pick_point, image)
    while True:
        cv2.imshow(WIN_NAME, image)
        key = cv2.waitKey(30)
        if key == 27:
            break
            cv2.destroyAllWindows()
    with open("img_txt.txt", "wb") as f:
        pickle.dump(txt_list, f)
        # f.writelines(txt_list)
        f.close()