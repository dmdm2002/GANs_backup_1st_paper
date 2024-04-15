import numpy as np
import time
import os
import copy
import sys
import glob as _glob
import csv
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import shutil
import pylib as py
import os
from torch.utils.data import Dataset
csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)

def csv2list(filename):
  lists=[]
  file=open(filename,"r")
  while True:
    line=file.readline().replace('\n','')
    if line:
      line=line.split(",")
      lists.append(line)
    else:
      break
  return lists

savepath='datasets/vein2vein/originimagetotrain(B)_with_origin/'

A_img_paths = py.glob(py.join('datasets', 'vein2vein', 'trainB_inter_class(only aug)'), '*/*',True) # origin image
B_img_paths = py.glob(py.join('generate_2fold(B)_2', '92epoch', 'train_A2B'), '*',True) # generated image

cslist=csv2list('generate_2fold(B)_2/all_92_img_ssim.csv')
ds_np = np.array(cslist)
for i,score in enumerate(ds_np[:,1]):
    targetpath=''
    if float(score)>0.3:
        targetpath = B_img_paths[i]
    else:
        targetpath = A_img_paths[i]

    split_imagepath = targetpath.split('\\')
    file = split_imagepath[len(split_imagepath)-1]
    file_folder = file[0:file.find('_')]
    py.mkdir(savepath + file_folder)
    shutil.copy(targetpath, savepath + file_folder + '/' + file)
    print(targetpath)
