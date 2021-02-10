# -*- coding: utf-8 -*-
# ---------------------


from conf import Conf
from dataset.dummy_ds import DataGenerator
from models.model import get_MobileCenterModel

import tensorflow as tf
from tensorflow import keras
from progress_bar import ProgressBar
from time import time
import numpy as np
import cv2

class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        self.cnf = cnf

        # init train loader
        self.train_loader = DataGenerator(self.cnf)
        self.test_loader = DataGenerator(self.cnf, partition='test', shuffle=False)

        # init model
        self.model = get_MobileCenterModel(input_shape=self.cnf.input_shape)

        # init optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.cnf.lr)

        # init loss
        self.loss_mse = tf.keras.losses.MeanSquaredError()
        # self.loss_mae = tf.keras.losses.MeanAbsoluteError()

        # init metrics
        self.test_mse_metric = keras.metrics.MeanSquaredError()

        # compile the model
        # self.model.compile(optimizer=self.optimizer,
        #                    loss={'hm': 'mse'},
        #                    metrics={'hm': 'mse'})

        # init logging stuffs

        # self.callbacks = [
        #     keras.callbacks.TensorBoard(log_dir=self.cnf.exp_log_path),
        #     keras.callbacks.ModelCheckpoint(self.cnf.exp_weights_path, verbose=1, save_best_only=True),
        #     keras.callbacks.EarlyStopping(patience=50, verbose=1, restore_best_weights=True)
        # ]

        self.log_path = cnf.exp_log_path
        print(f'tensorboard --logdir={cnf.project_log_path.abspath()}\n')
        self.sw = tf.summary.create_file_writer(self.log_path)
        self.log_freq = len(self.train_loader)
        self.train_losses = []

        # starting values
        self.epoch = 0
        self.best_test_loss = None

        # init progress bar
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

        # possibly load checkpoint
        self.load_ck()


    def load_ck(self):
        """
        load training checkpoint
        """

        if self.cnf.exp_weights_path.exists():
            # self.model.load_weights(latest)
            self.model = keras.models.load_model(self.cnf.exp_weights_path)
            print(f'[loaded checkpoint \'{self.cnf.exp_weights_path}\']')

    def save_ck(self, save_opt=True):
        """
        save training checkpoint
        """
        save_path = self.cnf.exp_weights_path if save_opt else self.cnf.exp_weights_path / "best"
        keras.models.save_model(self.model, save_path, include_optimizer=save_opt)


    def train(self):
        """
        train model for one epoch on the Training-Set.
        """

        # fit the model
        # self.model.fit(x=self.train_loader, epochs=self.cnf.epochs, validation_data=self.val_loader,
        #                callbacks=self.callbacks, use_multiprocessing=True, workers=self.cnf.n_workers)

        start_time = time()
        times = []
        # self.train_loader.on_epoch_end()
        for step, sample in enumerate(self.train_loader):
            t = time()

            x, y_true = sample

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                y_pred = self.model(x, training=True)

                # Compute the loss value for this minibatch.
                loss = self.loss_mse(y_pred, y_true)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss, self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            self.train_losses.append(loss)

            # print an incredible progress bar
            times.append(time() - t)
            if self.cnf.log_each_step or (not self.cnf.log_each_step and self.progress_bar.progress == 1):
                print(f'\r{self.progress_bar} '
                      f'│ Loss: {np.mean(self.train_losses):.6f} '
                      f'│ ↯: {1 / np.mean(times):5.2f} step/s', end='')
            self.progress_bar.inc()

        # log average loss of this epoch
        mean_epoch_loss = np.mean(self.train_losses)
        with self.sw.as_default():
            tf.summary.scalar(name='train_loss', data=mean_epoch_loss, step=self.epoch)
            self.sw.flush()
        self.train_losses = []

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')


    def test(self):
        """
        test model on the Test-Set
        """

        t = time()
        with self.sw.as_default():
            for step, sample in enumerate(self.test_loader):
                x, y_true = sample
                y_pred = self.model(x, training=False)

                self.test_mse_metric.update_state(y_true, y_pred)

                # draw results for this step in a 3 rows grid:
                # row #1: input (x)
                # row #2: predicted_output (y_pred)
                # row #3: target (y_true)

                if step == 0:
                    x_np_ = (x[0].copy() * 255).astype(np.uint8)
                    y_pred_np_ = y_pred[0].numpy()
                    cv2.normalize(y_pred_np_,y_pred_np_,0,255,cv2.NORM_MINMAX)
                    y_pred_np_ = np.array(y_pred_np_, dtype=np.uint8)
                    y_true_np_ = (y_true[0].copy()  * 255).astype(np.uint8)
                    x_np_ = np.expand_dims(cv2.resize(cv2.cvtColor(x_np_, cv2.COLOR_RGB2GRAY), (y_pred_np_.shape[1], y_pred_np_.shape[0])),2)
                    grid = np.expand_dims(np.hstack([x_np_, y_pred_np_, y_true_np_]),0)
                    tf.summary.image(name=f'results_{step}', data=grid, step=self.epoch)
                    self.sw.flush()

            # log average loss on test set
            mean_test_mse = self.test_mse_metric.result()
            print(f'\t● AVG MSE on TEST-set: {mean_test_mse:.6f} │ T: {time() - t:.2f} s')

            tf.summary.scalar(name='test_mse', data=mean_test_mse, step=self.epoch)
            self.sw.flush()

            # save best model
            if self.best_test_loss is None or mean_test_mse < self.best_test_loss:
                self.best_test_loss = mean_test_mse
                self.save_ck(save_opt=False)

    def export_tflite(self):
        """
        convert saved checkpoint to tflite
        """

        if self.cnf.exp_weights_path.exists():

            # Convert the model.
            converter = tf.lite.TFLiteConverter.from_saved_model(self.cnf.exp_weights_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT] # quant 8 bit

            def representative_dataset_gen():
                img_list = []
                for batch in self.train_loader:
                    # Get sample input data as a numpy array in a method of your choosing.
                    for img in batch[0]:
                        img_list.append(img)

                img = tf.data.Dataset.from_tensor_slices(img_list).batch(1)
                for i in img.take(self.cnf.batch_size):
                    yield [i]

            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8
            converter.inference_output_type = tf.int8  # or tf.uint8

            tflite_model = converter.convert()

            self.cnf.tflite_model_outpath.makedirs_p()

            with tf.io.gfile.GFile(self.cnf.tflite_model_outpath /'model.tflite', 'wb') as f:
                f.write(tflite_model)




    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for _ in range(self.epoch, self.cnf.epochs):
            self.train()

            self.test()

            self.epoch += 1
            self.save_ck()

        if self.cnf.export_tflite:
            self.export_tflite()