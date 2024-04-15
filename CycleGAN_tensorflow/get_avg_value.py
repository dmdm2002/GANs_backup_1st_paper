from tensorflow import image as tim
import tensorflow as tf
import os
import sys
import csv
import timeit
import numpy as np

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import module
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
csv.register_dialect(
    'mydialect',
    delimiter=',',
    quotechar='"',
    doublequote=True,
    skipinitialspace=True,
    lineterminator='\r\n',
    quoting=csv.QUOTE_MINIMAL)
# ==============================================================================
# =                                   param                                    =
# ==============================================================================



py.arg('--experiment_dir', default='generate_2fold(B)_2')
py.arg('--batch_size', type=int, default=24)
test_args = py.args()

args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

# output_dir
output_dir = py.join('output_interclass_shuffle_2fold(B)', args.dataset)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

SHUFFLE_BUFFER_SIZE = 8
numberofclass = 301

created_path = 'generate_2fold(B)_2/'

originpath = 'dataset/vein2vein/trainB_inter_class(only aug)/'

A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainB_inter_class(only aug)'), '*/*', True)

A_dataset_test = data.make_dataset(A_img_paths, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)

meanscores = []
avgs1=[]
avgs2=[]
avgs3=[]
for i,images in enumerate(A_dataset_test):
    for j in range(3):
        orimean = tf.reduce_mean(images[:, :, :, j])
        if j==0:
            avgs1.append(orimean.numpy())
        elif j==1:
            avgs2.append(orimean.numpy())
        else:
            avgs3.append(orimean.numpy())

avg1=np.mean(avgs1)
avg2=np.mean(avgs2)
avg3=np.mean(avgs3)
f = open('generate_2fold(B)_2/origin_image_pixel_AVG.csv', 'a', newline='')
wr = csv.writer(f)
wr.writerow(['ch1', avg1])
wr.writerow(['ch2', avg2])
wr.writerow(['ch3', avg3])

wr.writerow(['ch1_norm', avg1*127.5+127.5])

wr.writerow(['ch1_norm', avg2*127.5+127.5])

wr.writerow(['ch1_norm', avg3*127.5+127.5])
f.close()
