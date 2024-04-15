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


py.arg('--experiment_dir', default='D:/cyclegan/output_interclass_shuffle_2fold(B)_1/realDB/')
py.arg('--batch_size', type=int, default=24)
test_args = py.args()

args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

# output_dir
output_dir = py.join('D:/cyclegan/output_interclass_shuffle_2fold(B)_1', args.dataset)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

SHUFFLE_BUFFER_SIZE = 8

created_path = 'D:/cyclegan/output_interclass_shuffle_2fold(B)_1/realDB/'

A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset), '*/*', True)
print(A_img_paths)
A_img_paths.sort(key=lambda x:len(x))
A_dataset_test = data.make_dataset(A_img_paths, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)


for i in range(179,200):
    path_per_epoch = created_path + str(i) + 'epoch'

    B_img_paths = py.glob(py.join(path_per_epoch,'train_A2B'), '*')
    B_img_paths.sort(key=lambda x: len(x))
    B_dataset_test = data.make_dataset(B_img_paths, args.batch_size, args.load_size, args.crop_size,
                                       training=False, drop_remainder=False, shuffle=False, repeat=1)


    zipeed_dataset = tf.data.Dataset.zip((A_dataset_test, B_dataset_test))

    scores = []
    chk = 0
    idx=0
    epoch_start = timeit.default_timer()
    for n, (input_image, target) in zipeed_dataset.enumerate():
        chk += 1
        '''
        im1 = tf.io.read_file(input_image)
        im1 = im.decode_bmp(im1, 3)
        im1 = resize(im1)

        im2 = tf.io.read_file(target)
        im2 = im.decode_bmp(im2, 3)

        # Compute SSIM over tf.float32 Tensors.
        im1 = im.convert_image_dtype(input_image, tf.float32)
        im2 = im.convert_image_dtype(target, tf.float32)
         '''
        ssim2 = tim.ssim(input_image, target, max_val=1.0, filter_size=11,
                        filter_sigma=1.5, k1=0.01, k2=0.03)
        for sc in ssim2:
            name = py.name(A_img_paths[idx])
            f = open('D:/cyclegan/output_interclass_shuffle_2fold(B)_1/181_200epoch_img_ssim.csv', 'a', newline='')
            wr = csv.writer(f)
            wr.writerow([name,sc.numpy()])
            f.close()
            idx+=1
        '''
        ssim2 = tim.ssim(input_image, target, max_val=1.0, filter_size=11,
                        filter_sigma=1.5, k1=0.01, k2=0.03)

        scores.append(ssim2.numpy())
        if chk > 505:
            print(chk)
        if chk == 507:
            # socres  [234,52] 이기 떄문에 2번 계산해줌
            mean = sum(scores, 0.0) / len(scores)
            mean = sum(mean, 0.0) / len(mean)
            meanscores.append(['EPOCH' + str(i), mean])
            f = open('generate_2fold(B)/ssim.csv', 'a', newline='')
            wr = csv.writer(f)
            wr.writerow(meanscores[i-300])
            f.close()
            epoch_end = timeit.default_timer()
            print(str(i)+' EPOCH/SEC: ', epoch_end - epoch_start)
        '''
##52 234
##24 507
##12 1014
##8  1521


# 936 52  18