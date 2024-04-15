import glob
import cv2
import numpy as np
import os
import sys
import shutil
import pylib as py
path='datasets/vein2vein/trainA_inter_class(only aug)/'
savepath='datasets/vein2vein/sobel_edge_origin(A)/'


imagePaths = py.glob(path, '*/*')

for imagePath in imagePaths:
    print(imagePath)
    split_imagepath = imagePath.split('/')

    if len(split_imagepath)>3:
        file=split_imagepath[3]
    else:
        file = imagePath.split('\\')
        split_imagepath = imagePath.split('\\')
    if len(file)>3:
        file_folder=file[0:file.find('_')]
    else:
        file_folder=file[len(file)-2]
    py.mkdir(savepath+file_folder)

    fullfilename=split_imagepath[len(split_imagepath)-1]

    image = cv2.imread(imagePath, 0)

    edges=cv2.Sobel(ksize=3,src=image,ddepth=-1,dx=3,dy=3,delta=127)

    #edges=cv2.Canny(image,0,40)

    print(edges)
    cv2.imwrite(savepath+file_folder+'/'+fullfilename, edges)

    #shutil.copy(imagePath,savepath+'/'+file_folder+'/'+file)
    print(imagePath)