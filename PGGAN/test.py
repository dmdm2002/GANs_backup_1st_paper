import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot
from math import sqrt
from PIL import Image
import os

from pgan import PGAN, WeightedSum
from tensorflow.keras import backend
import os


def on_epoch_end(self, epoch, logs=None):
    samples = self.model.generator(self.random_latent_vectors)
    samples = (samples * 0.5) + 0.5
    n_grid = int(sqrt(self.num_img))

    fig, axes = pyplot.subplots(n_grid, n_grid, figsize=(4 * n_grid, 4 * n_grid))
    sample_grid = np.reshape(samples[:n_grid * n_grid],
                             (n_grid, n_grid, samples.shape[1], samples.shape[2], samples.shape[3]))

    for i in range(n_grid):
        for j in range(n_grid):
            axes[i][j].set_axis_off()
            samples_grid_i_j = Image.fromarray((sample_grid[i][j] * 255).astype(np.uint8))
            samples_grid_i_j = samples_grid_i_j.resize((128, 128))
            axes[i][j].imshow(np.array(samples_grid_i_j))
    title = f'Z:/Iris_dataset/nd_labeling_iris_data/PGGAN/1-fold/sample/plot_{self.prefix}_{epoch:05d}.png'
    pyplot.savefig(title, bbox_inches='tight')
    print(f'\n saved {title}')
    pyplot.close(fig)


DATA_ROOT = 'Z:/Iris_dataset/nd_labeling_iris_data/PGGAN/1-fold/A'
NOISE_DIM = 512
# Set the number of batches, epochs and steps for trainining.
# Look 800k images(16x50x1000) per each lavel
BATCH_SIZE = [1, 1, 1, 1, 1, 1, 1]
EPOCHS = 40
STEPS_PER_EPOCH = 2277

def preprocessing_image(img):
  img = img.astype('float32')
  img = (img - 127.5) / 127.5
  return img

train_image_generator = ImageDataGenerator(horizontal_flip=True, preprocessing_function=preprocessing_image)

steps_per_epoch = STEPS_PER_EPOCH
DATA_ROOT = f'Z:/Iris_dataset/nd_labeling_iris_data/PGGAN/1-fold/A'
train_dataset = train_image_generator.flow_from_directory(batch_size=1,
                                                          directory=DATA_ROOT,
                                                          shuffle=True,
                                                          target_size=(256, 256),
                                                          class_mode='binary')

pgan = PGAN(
    latent_dim = NOISE_DIM,
    d_steps = 7,
)
generator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

pgan.stabilize_generator()

pgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
)

checkpoint_path = f"Z:/Iris_dataset/nd_labeling_iris_data/PGGAN/1-fold/checkpoint/pgan_{cbk.prefix}.ckpt"
pgan.load_weights(checkpoint_path)

pgan.test_on_batch(train_dataset, sample_weight=checkpoint_path)