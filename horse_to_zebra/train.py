import time

import tensorflow as tf


class TrainModel:
    def __init__(self, settings, load_data_instance, loss_function_instance):
        self.epochs = settings.EPOCHS
        self.generator_g = settings.generator_g
        self.generator_f = settings.generator_f
        self.generator_f_optimizer = settings.generator_f_optimizer
        self.generator_g_optimizer = settings.generator_g_optimizer
        self.discriminator_x_optimizer = settings.discriminator_x_optimizer
        self.discriminator_y_optimizer = settings.discriminator_y_optimizer
        self.discriminator_x = settings.discriminator_x
        self.discriminator_y = settings.discriminator_y
        self.train_horses, self.train_zebras, _, _ = load_data_instance.map_data()
        self.loss_function_instance = loss_function_instance
        self.checkpoint_dir = "./checkpoints"
        self.checkpoint = tf.train.Checkpoint(
            generator_g=self.generator_g,
            generator_f=self.generator_f,
            discriminator_x=self.discriminator_x,
            discriminator_y=self.discriminator_y,
            generator_g_optimizer=self.generator_g_optimizer,
            generator_f_optimizer=self.generator_f_optimizer,
            discriminator_x_optimizer=self.discriminator_x_optimizer,
            discriminator_y_optimizer=self.discriminator_y_optimizer,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=5
        )

    def train_step(self, real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.loss_function_instance.generator_loss(disc_fake_y)
            gen_f_loss = self.loss_function_instance.generator_loss(disc_fake_x)

            total_cycle_loss = self.loss_function_instance.calc_cycle_loss(
                real_x, cycled_x
            ) + self.loss_function_instance.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = (
                gen_g_loss
                + total_cycle_loss
                + self.loss_function_instance.identity_loss(real_y, same_y)
            )

            total_gen_f_loss = (
                gen_f_loss
                + total_cycle_loss
                + self.loss_function_instance.identity_loss(real_x, same_x)
            )

            disc_x_loss = self.loss_function_instance.discriminator_loss(
                disc_real_x, disc_fake_x
            )
            disc_y_loss = self.loss_function_instance.discriminator_loss(
                disc_real_y, disc_fake_y
            )

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(
            total_gen_g_loss, self.generator_g.trainable_variables
        )
        generator_f_gradients = tape.gradient(
            total_gen_f_loss, self.generator_f.trainable_variables
        )

        discriminator_x_gradients = tape.gradient(
            disc_x_loss, self.discriminator_x.trainable_variables
        )
        discriminator_y_gradients = tape.gradient(
            disc_y_loss, self.discriminator_y.trainable_variables
        )

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(
            zip(generator_g_gradients, self.generator_g.trainable_variables)
        )

        self.generator_f_optimizer.apply_gradients(
            zip(generator_f_gradients, self.generator_f.trainable_variables)
        )

        self.discriminator_x_optimizer.apply_gradients(
            zip(discriminator_x_gradients, self.discriminator_x.trainable_variables)
        )

        self.discriminator_y_optimizer.apply_gradients(
            zip(discriminator_y_gradients, self.discriminator_y.trainable_variables)
        )
        print("inside train_step")

    def train_loop(self):
        for epoch in range(self.epochs):
            print("inside train_loop")
            start = time.time()
            n = 0
            for image_x, image_y in tf.data.Dataset.zip(
                (self.train_horses, self.train_zebras)
            ):
                self.train_step(image_x, image_y)
                if n % 10 == 0:
                    print(".", end="")
                n += 1
            print("finished loop")
            self.checkpoint_manager.save()
            print(
                "Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, time.time() - start
                )
            )
        if self.checkpoint.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print("Latest checkpoint restored!")
