# according to https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/pix2pix.ipynb
import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
import numpy as np

PATH = 'Data/tmp_data_magmap/'
print(os.listdir(PATH))

sample_image = np.squeeze(np.load(PATH + 'train/magmap_1800_1_45.npy')[0, :, :, :])

plt.figure()
plt.imshow(sample_image / 100 + .5)
plt.gca().invert_yaxis()
plt.title('rtp(RNT) to RGB')
plt.xlabel('lon [pixel]')
plt.ylabel('lat [pixel]')
plt.show()


def load(file_name):
    # print('----------load start-----------')
    # print('file_name: ', file_name)
    if type(file_name) != str:
        file_name = file_name.numpy()
        with open('loadedfiles.txt', 'a') as f:
            f.write(str(file_name))
    magmap_pair = np.load(file_name, allow_pickle=True)
    # print('magmap_pair: ', magmap_pair)
    input_image = np.squeeze(magmap_pair[0, :, :, :])  # / 100 + 0.5
    real_image = np.squeeze(magmap_pair[1, :, :, :])  # / 0.4 + 0.5

    input_image[:, :, 0] = input_image[:, :, 0] / 100. + .5
    input_image[:, :, 1] = input_image[:, :, 1] / 50. + .5
    input_image[:, :, 2] = input_image[:, :, 2] / 50. + .5

    real_image[:, :, 0] = real_image[:, :, 0] / .4 + .5
    real_image[:, :, 1] = real_image[:, :, 1] / .2 + .5
    real_image[:, :, 2] = real_image[:, :, 2] / .2 + .5

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    # print('input_image: ', input_image)
    # print('real_image: ', real_image)
    # print('----------load end-----------')
    return input_image, real_image


print('--------visualize sample data---------')
train_dataset = tf.data.Dataset.list_files(PATH + 'train/magmap_1800_1_45.npy')
tf.data.Dataset.list_files(PATH + 'train/magmap_1800_1_45.npy')
inp, re = load(PATH + 'train/magmap_1800_1_45.npy')
# Casting to int for matplotlib to display the images
plt.figure()
plt.imshow(inp)
plt.figure()
plt.imshow(re)
plt.show()
print('--------visualize sample data end---------')

# The facade training set consist of 400 images
BUFFER_SIZE = 4
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 128


def resize(input_image, real_image, height, width):
    # print('---------resize start--------')
    # print(input_image)
    input_image.set_shape([128, 256, 3])
    real_image.set_shape([128, 256, 3])
    # print(input_image)
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # print('--------resize end----------')
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    # print('----------normalize start----------')
    norm_max = tf.reduce_max([abs(tf.reduce_max(input_image)), abs(tf.reduce_min(input_image))])
    # print('maxmium value for normalization: ', norm_max)
    input_image = (input_image / norm_max)
    real_image = (real_image / norm_max)
    # print('---------normalize end-------------')
    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 140, 280)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


plt.figure(figsize=(6, 6))
for i in range(4):
    rj_inp, rj_re = random_jitter(inp, re)
    plt.subplot(2, 2, i + 1)
    plt.imshow(rj_inp)
    plt.axis('off')
    plt.title('random jitter')
plt.show()


def load_image_train(image_file):
    # print('--------load image train start--------')
    # print('image_file', image_file)
    [input_image, real_image, ] = tf.py_function(func=load, inp=[image_file], Tout=[tf.float32, tf.float32])
    # print(X)
    # input_image, real_image = X
    # print('input_image: ', input_image)
    # print('real_image: ', real_image)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    # print('---------load iamge train end---------')
    return input_image, real_image


def load_image_test(image_file):
    [input_image, real_image] = tf.py_function(load, inp=[image_file], Tout=[tf.float32, tf.float32])
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


PATH = 'Data/tmp_data_magmap/'
# train_file_list = os.listdir('Data/tmp_data_magmap/train/*.npy')
# train_file_list = [PATH + 'train/' + fn for fn in train_file_list]
# train_dataset = tf.data.Dataset.from_tensor_slices(train_file_list)
train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.npy')
train_dataset = train_dataset.map(load_image_train)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

try:
    test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.npy')
except tf.errors.InvalidArgumentError:
    test_dataset = tf.data.Dataset.list_files(PATH + 'val/*.npy')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print(down_result.shape)


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


up_model = upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)


def Generator():
    inputs = tf.keras.layers.Input(shape=[128, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64) (180,90)
        downsample(128, 4),  # (batch_size, 64, 64, 128) (90,45)
        downsample(256, 4),  # (batch_size, 32, 32, 256) (45,23)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        # downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]
    up_stack = [
        # upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        # print(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        # print(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


print('--------test generator--------')
generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=128)

gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])
plt.show()

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[128, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[128, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


print('--------test discriminator----------')
discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=128)

disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()
plt.show()


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


print('-----------set optimizer & checkpoint---------')
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    plt.close()


for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)

log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
        tf.summary.scalar('disc_loss', disc_loss, step=step)


def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 100 == 0:
            # display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 100 steps: {time.time() - start:.2f} sec\n')

            start = time.time()

            generate_images(generator, example_input, example_target)
            print(f"Step: {step // 100}00")

        train_step(input_image, target, step)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 100 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# %load_ext tensorboard
# %tensorboard --logdir {log_dir}

fit(train_dataset, test_dataset, steps=5000)
# for example_input, example_target in test_dataset.take(1):
#     generate_images(generator, example_input, example_target)
