# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:58:08 2019

@author: Administrator
"""

import cv2, math, re
import numpy as np
import matplotlib.pyplot as plt

# UTM坐标
'''
p1 = [650303.725, 3291302.209];
p2 = [650290.429, 3291310.085];
p3 = [650306.92, 3291307.567];
p4 = [650293.655, 3291315.601];
p5 = [650300.524, 3291311.537];
p6 = [650296.828, 3291306.255];
# p7 = [650028.683, 3291704.701];
# p8 = [650029.631, 3291700.284];
po = np.array([650291.285, 3291296.851]);
'''
p1 = [650287.373, 3291290.626];
p2 = [650293.737, 3291286.141];
p3 = [650283.133, 3291270.95];
p4 = [650278.351, 3291274.428];
p5 = [650261.767, 3291257.283];
p6 = [650267.037, 3291254.407];
p7 = [650256.603, 3291240.868];
p8 = [650248.066, 3291240.261];
p9 = [650245.245, 3291241.917];
p10 = [650235.147, 3291219.062];
p11 = [650229.406, 3291224.302]
po = np.array([650287.373, 3291290.626]);

# 雷达坐标
'''
r1 = [-334, 352];
r2 = [-343, 1799];
r3 = [300, 317];
r4 = [408, 1823];
r5 = [328, 1009];
r6 = [-315, 1077];
# r7 = [-209, 1171];
# r8 = [193, 1171];
'''
r1 = [400, 2089];
r2 = [-217, 2199];
r3 = [-278, 4046];
r4 = [283, 4046];
r5 = [748, 6279];
r6 = [182, 6303];
r7 = [313, 7956];
r8 = [959, 8490];
r9 = [1367, 8452];
r10 = [856, 11046];
r11 = [1747, 10848]

dst = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9]) # WGS84坐标系（m）
# dst = np.array([p1,p2,p3,p4,p5,p6]) # WGS84坐标系（m）
dst = (dst-po)*100;
src = np.array([r1,r2,r3,r4,r5,r6,r7,r8,r9]) # 雷达坐标系(cm); 
# src = np.array([r1,r2,r3,r4,r5,r6]) # 雷达坐标系(cm); 
(h1, status) = cv2.findHomography(src, dst) # h1为雷达平面坐标系到世界坐标系的单应性矩阵

plt.figure();
plt.clf()
# (x, y) = (dst[:,0], dst[:,1]);
# plt.plot(x, y)

with open('D:/location6_south/south.txt' , 'r') as fr:
    Lines = fr.readlines()
fr.close()
Lanes = list()
for i in range(len(Lines)):
    line = re.split(', ', Lines[i].strip());
    line[0] = float(line[0])
    line[1] = float(line[1])
    Lanes.append(line)
Lanes = np.array(Lanes)
(x, y) = (Lanes[:,0], Lanes[:,1])
plt.scatter(x,y, s=10)
    
'''
Lr = np.empty((0,2))
for i in range(6):
    (Xp, Yp) = (src[i,0],src[i,1]); # (Xp, Yp) = (rt[0],rt[1])
    Pp = np.array([[Xp],[Yp],[1]]) # 图像齐次坐标系
    tmp = np.dot(h1,Pp)
    Pw = tmp[0:2]/tmp[-1]
    # print(Pw)
    Lr = np.vstack((Lr,Pw.T/100));
print (Lr * 100)

target_UTM = Lr + po
for i in range(target_UTM.shape[0]):
    (x, y) = (target_UTM[i,0], target_UTM[i,1])
    plt.scatter(x,y, color = 'g', s=15)
    plt.pause(2)
    
'''
rotation_degree = np.empty((0,1));
sincos_theta = np.empty((0,2));
dst = np.array([p1,p2,p3,p4,p5,p6,p7,p8])
for j in range(dst.shape[0]):
    dst = np.array([p1,p2,p3,p4,p5,p6,p7,p8])
    src = np.array([r1,r2,r3,r4,r5,r6,r7,r8])
    dst = (dst-dst[j])*100;src = (src-src[j]);
    for i in range(dst.shape[0]):
        (x,y) = (src[i,0],src[i,1]);
        (Xw, Yw) = (dst[i,0],dst[i,1]);
        if [x,y] != [0,0]:
            cos_theta = (x*Xw + y*Yw)/(x*x + y*y)
            sin_theta = (y*Xw - x*Yw)/(x*x + y*y)
            Norm = (cos_theta**2 + sin_theta**2)**0.5
            cos_theta = cos_theta/Norm
            sin_theta = sin_theta/Norm
            if sin_theta >= 0 and cos_theta >=0:
                rotation_degree1 = math.degrees(math.acos(cos_theta)) # 计算结果
                rotation_degree2 = math.degrees(math.asin(sin_theta))
            elif sin_theta >= 0 and cos_theta < 0:
                rotation_degree1 = math.degrees(math.acos(cos_theta)) # 计算结果
                rotation_degree2 = 180 - math.degrees(math.asin(sin_theta))
            elif sin_theta < 0 and cos_theta >=0:
                rotation_degree1 = 360 - math.degrees(math.acos(cos_theta))
                rotation_degree2 = 360 + math.degrees(math.asin(sin_theta))
            elif sin_theta < 0 and cos_theta < 0:
                rotation_degree1 = 360 - math.degrees(math.acos(cos_theta))
                rotation_degree2 = 180 - math.degrees(math.asin(sin_theta))               
            rotation_degree = np.vstack((rotation_degree, (rotation_degree1+rotation_degree2)/2))
            sincos_theta = np.vstack((sincos_theta, [sin_theta, cos_theta]))           
print (rotation_degree)
print (sincos_theta);

rotation_degree = np.mean(rotation_degree);
sincos_theta = np.mean(sincos_theta, axis = 0)

cos_theta = math.cos(math.radians(rotation_degree))
sin_theta = math.sin(math.radians(rotation_degree))

#cos_theta = sincos_theta[1]
#sin_theta = sincos_theta[0]
#Norm = (cos_theta**2 + sin_theta**2)**0.5
#cos_theta = cos_theta/Norm
#sin_theta = sin_theta/Norm

R_c2w = np.array([[cos_theta, sin_theta],
                  [-sin_theta, cos_theta]]);
R_w2c = np.linalg.inv(R_c2w);

'''    
p1 = [650303.725, 3291302.209];
p2 = [650290.429, 3291310.085];
p3 = [650306.92, 3291307.567];
p4 = [650293.655, 3291315.601];
p5 = [650300.524, 3291311.537];
p6 = [650296.828, 3291306.255];
# p7 = [650028.683, 3291704.701];
# p8 = [650029.631, 3291700.284];
po = np.array([650291.285, 3291296.851]);
'''
p1 = [650287.373, 3291290.626];
p2 = [650293.737, 3291286.141];
p3 = [650283.133, 3291270.95];
p4 = [650278.351, 3291274.428];
p5 = [650261.767, 3291257.283];
p6 = [650267.037, 3291254.407];
p7 = [650256.603, 3291240.868];
p8 = [650248.066, 3291240.261];
p9 = [650245.245, 3291241.917];
p10 = [650235.147, 3291219.062];
p11 = [650229.406, 3291224.302]
po = np.array([650287.373, 3291290.626]);

'''
c1 = [0.86979, 0.55926];
c2 = [0.26354, 0.59907];
c3 = [0.79167, 0.4963];
c4 = [0.25833, 0.53148];
c5 = [0.54167, 0.51296];
c6 = [0.5724, 0.58426];
'''
c1 = [0.64844, 0.61111];
c2 = [0.42865, 0.60833];
c3 = [0.47708, 0.4537];
c4 = [0.5724, 0.45278];
c5 = [0.59635, 0.38333];
c6 = [0.53906, 0.38241];
c7 = [0.54583, 0.35833];
c8 = [0.59479, 0.35370];
c9 = [0.61875, 0.35463];
c10 = [0.57343, 0.33518];
c11 = [0.62083, 0.33519];

# co = np.array(c1);
dst = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9]) # WGS84坐标系（m）
dst = (dst-po)*100; #(cm)
src = np.array([c1,c2,c3,c4,c5,c6,c7,c8,c9]); # 图像坐标系
(h2, status) = cv2.findHomography(src, dst) # h2为图像坐标系到世界坐标系的单应性矩阵
'''
Lc = np.empty((0,2))
for i in range(9):
    (Xp, Yp) = (src[i,0],src[i,1]);
    Pp = np.array([[Xp],[Yp],[1]]) # 图像齐次坐标系
    tmp = np.dot(h2,Pp)
    Pw = tmp[0:2]/tmp[-1]
    # print(Pw);
    Lc = np.vstack((Lc,Pw.T/100));
print(Lc*100.)

target_UTM = Lc + po
for i in range(target_UTM.shape[0]):
    (x, y) = (target_UTM[i,0], target_UTM[i,1])
    plt.scatter(x,y, color = 'r', s=15)
    plt.pause(2)

'''
