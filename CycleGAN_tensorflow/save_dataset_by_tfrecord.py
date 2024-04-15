############################
# tfrecord 파일저장 및 LOAD #
#############################
import numpy as np
import tensorflow as tf
import os
origin_ds_path='datasets/vein2vein/origin_image_blur(B)/'


#region functions
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#endregion

#Hyper parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224

#Hyper parameters
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

#이미지 데이터셋 list 배열
list_ds = tf.data.Dataset.list_files(str(origin_ds_path+'*/*'),shuffle=False)

#class(분류될 종류) 배열
CLASS_NAMES=os.listdir(origin_ds_path)
print(CLASS_NAMES)

def image_example(image_string, label):
  image_shape = tf.image.decode_bmp(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

record_file='images_Bset_blur.tfrecord'

with tf.io.TFRecordWriter(record_file) as writer:
  for i, d in enumerate(list_ds):
    fname = d.numpy().decode('utf-8')
    cls=int(fname.split('\\')[-2])
    filename = str(fname.split('\\')[-1])
    image_string = open(fname, 'rb').read()
    tf_example=image_example(image_string,cls)
    writer.write(tf_example.SerializeToString())

'''
for i,d in enumerate(list_ds):
  fname=d.numpy().decode('utf-8')
  cls=str(fname.split('\\')[-2])
  image_string=open(fname,'rb').read()
  print(image_string)

  break
'''

