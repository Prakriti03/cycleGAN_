import tensorflow as tf


def load_and_preprocess_image(file_path):
    # Read and decode the image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = normalize(image)
    return image


def normalize(image):
    # normalizing the images to [-1, 1]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image
