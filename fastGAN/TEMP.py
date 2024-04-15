import shutil
import os

path = 'D:/ND/original/B'
temp = os.listdir(path)
print(temp)

OUTPUT = 'D:/ND/original/B_live'
os.makedirs(OUTPUT, exist_ok=True)

for folder in temp:
    img_path = f'{path}/{folder}'
    img_names = os.listdir(img_path)

    for name in img_names:
        source = f'{path}/{folder}/{name}'
        destination = f'{OUTPUT}/{name}'
        shutil.copy(source, destination)