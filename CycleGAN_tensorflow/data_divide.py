import shutil
import pylib as py
import os
path='generate_2fold(B)/300epoch/train_A2B/'
savepath='datasets/vein2vein/originimagetotrain(B)_1/'
py.mkdir(savepath)
py.mkdir(savepath+'/001')
imagePaths = [os.path.join(path, file_name) for file_name in os.listdir(path)]
for imagePath in imagePaths:
    split_imagepath = imagePath.split('/')
    file = split_imagepath[3]
    file_folder=file[0:file.find('_')]
    py.mkdir(savepath+file_folder)
    shutil.copy(imagePath,savepath+file_folder+'/'+file)
    print(imagePath)