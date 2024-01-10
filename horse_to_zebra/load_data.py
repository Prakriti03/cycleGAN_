import tensorflow as tf
from preprocess import load_and_preprocess_image


class LoadData:
    def __init__(self, settings):
        self.settings = settings

    def load_data(self):
        path_to_train_horse_file = self.settings.train_horses_path
        path_to_train_zebra_file = self.settings.train_zebras_path
        path_to_test_horse_file = self.settings.test_horses_path
        path_to_test_zebra_file = self.settings.test_zebras_path

        train_horses_image_path = tf.data.Dataset.list_files(
            path_to_train_horse_file + "/*.jpg", shuffle=True
        )
        train_zebras_image_path = tf.data.Dataset.list_files(
            path_to_train_zebra_file + "/*.jpg", shuffle=True
        )
        test_horses_image_path = tf.data.Dataset.list_files(
            path_to_test_horse_file + "/*.jpg", shuffle=True
        )
        test_zebras_image_path = tf.data.Dataset.list_files(
            path_to_test_zebra_file + "/*.jpg", shuffle=True
        )

        return (
            train_horses_image_path,
            train_zebras_image_path,
            test_horses_image_path,
            test_zebras_image_path,
        )

    def map_data(self):
        (
            train_horses_image_path,
            train_zebras_image_path,
            test_horses_image_path,
            test_zebras_image_path,
        ) = self.load_data()

        train_horses = (
            train_horses_image_path.map(
                load_and_preprocess_image,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .shuffle(self.settings.BUFFER_SIZE)
            .batch(self.settings.BATCH_SIZE)
        )
        train_zebras = (
            train_zebras_image_path.map(
                load_and_preprocess_image,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .shuffle(self.settings.BUFFER_SIZE)
            .batch(self.settings.BATCH_SIZE)
        )
        test_horses = (
            test_horses_image_path.map(
                load_and_preprocess_image,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .shuffle(self.settings.BUFFER_SIZE)
            .batch(self.settings.BATCH_SIZE)
        )
        test_zebras = (
            test_zebras_image_path.map(
                load_and_preprocess_image,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .shuffle(self.settings.BUFFER_SIZE)
            .batch(self.settings.BATCH_SIZE)
        )

        return train_horses, train_zebras, test_horses, test_zebras
