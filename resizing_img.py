import os
import glob
import cv2

path = f'E:/backup/StyleGAN/ND/1-fold/test'

original_path = f'{path}/i_s'
resizing_path = f'{path}/resizing'

imgNames = os.listdir(original_path)
os.mkdir(resizing_path)

for Name in imgNames:
    path = f'{original_path}/{Name}'
    img = cv2.imread(path)
    img = cv2.resize(img, dsize=(224, 224), interpolation='')
    cv2.imwrite(f'{resizing_path}/{Name}', img)