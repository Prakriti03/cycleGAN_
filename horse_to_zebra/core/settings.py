import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


class Settings:
    BUFFER_SIZE = 100
    BATCH_SIZE = 1
    IMAGE_SIZE = (256, 256)
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    LAMBDA = 10
    EPOCHS = 10
    OUTPUT_CHANNELS = 3

    train_horses_path = (
        r"C:\Users\Dell\Desktop\naami\Datasets\horse-to-zebra-dataset\train_A_resampled"
    )
    train_zebras_path = (
        r"C:\Users\Dell\Desktop\naami\Datasets\horse-to-zebra-dataset\train_B_resampled"
    )
    test_horses_path = (
        r"C:\Users\Dell\Desktop\naami\Datasets\horse-to-zebra-dataset\testA"
    )
    test_zebras_path = (
        r"C:\Users\Dell\Desktop\naami\Datasets\horse-to-zebra-dataset\testB"
    )

    generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type="instancenorm")
    generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type="instancenorm")
    discriminator_x = pix2pix.discriminator(norm_type="instancenorm", target=False)
    discriminator_y = pix2pix.discriminator(norm_type="instancenorm", target=False)

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
