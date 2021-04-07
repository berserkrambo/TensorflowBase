# -*- coding: utf-8 -*-
# ---------------------


from conf import Conf
from dataset.widerface import DataGenerator
from models.model import get_MobileCenterModel

import tensorflow as tf
from tensorflow import keras
from progress_bar import ProgressBar
from time import time
import numpy as np
import cv2
import torch
import torchvision as tv

class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        self.cnf = cnf

        # init train loader
        self.train_loader = DataGenerator(self.cnf, partition='train', shuffle=True)
        self.test_loader = DataGenerator(self.cnf, partition='test', shuffle=False)

        # init model
        self.model = get_MobileCenterModel(input_shape=self.cnf.input_shape)

        # init optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.cnf.lr)

        # init loss
        self.loss_mse = tf.keras.losses.MeanSquaredError()
        # self.loss_mae = tf.keras.losses.MeanAbsoluteError()

        # init metrics
        self.test_h_mse_metric = keras.metrics.MeanSquaredError()
        self.test_cnt_mse_metric = keras.metrics.MeanSquaredError()
        self.test_s_mse_metric = keras.metrics.MeanSquaredError()

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
            y_true_h, y_true_cnt, y_true_sz = y_true
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                y_pred_h, y_pred_cnt, y_pred_s = self.model(x, training=True)

                # Compute the loss value for this minibatch.
                loss_h = self.loss_mse(y_true_h, y_pred_h)
                loss_cnt = self.loss_mse(y_true_cnt, y_pred_cnt)
                loss_s = self.loss_mse(y_true_sz, y_pred_s)

                loss = loss_h + loss_cnt + loss_s

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss, self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            self.train_losses.append(loss)

            if step == 0:
                hm = y_pred_h.numpy()
                hm = torch.nn.functional.max_pool2d(hm, kernel_size=5, padding=2, stride=1)
                hm[hm <= 0.7] = 0.0
                hm[hm > 1.0] = 1.0
                hm *= 255

                to_show = []
                x = x.cpu().numpy().transpose(0, 2, 3, 1)
                for i, hi in enumerate(hm.squeeze().cpu().numpy().astype(np.uint8)):
                    hii = np.stack([hi, hi, hi], axis=2)
                    hii = cv2.resize(hii, (self.cnf.input_shape[0], self.cnf.input_shape[0]))

                    hii = cv2.applyColorMap(hii, cv2.COLORMAP_JET)

                    hii = cv2.addWeighted((x[i] * 255).astype(np.uint8), 0.6, hii, 0.4, 0)

                    to_show.append(np.asarray(hii, dtype=np.float32) / 255.0)

                to_show = torch.from_numpy(np.asarray(to_show, dtype=np.float32).transpose(0, 3, 1, 2))
                grid = tv.utils.make_grid(to_show, normalize=True, range=(0, 1), nrow=x.shape[0])
                grid = (grid.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                tf.summary.image(name=f'results_{step}', data=grid, step=self.epoch)
                self.sw.flush()

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
                y_true_h, y_true_cnt, y_true_sz = y_true
                y_pred_h, y_pred_cnt, y_pred_s = self.model(x, training=False)

                self.test_h_mse_metric.update_state(y_true_h, y_pred_h)
                self.test_cnt_mse_metric.update_state(y_true_cnt, y_pred_cnt)
                self.test_s_mse_metric.update_state(y_true_sz, y_pred_s)

                # draw results for this step in a 3 rows grid:
                # row #1: input (x)
                # row #2: predicted_output (y_pred)
                # row #3: target (y_true)

                if step == 0:
                    hm = torch.nn.functional.max_pool2d(hm, kernel_size=5, padding=2, stride=1)
                    hm[hm <= 0.7] = 0.0
                    hm[hm > 1.0] = 1.0
                    hm *= 255

                    to_show = []
                    x = x.cpu().numpy().transpose(0, 2, 3, 1)
                    for i, hi in enumerate(hm.squeeze().cpu().numpy().astype(np.uint8)):
                        hii = np.stack([hi, hi, hi], axis=2)
                        hii = cv2.resize(hii, (self.cnf.input_shape[0], self.cnf.input_shape[0]))

                        hii = cv2.applyColorMap(hii, cv2.COLORMAP_JET)

                        hii = cv2.addWeighted((x[i] * 255).astype(np.uint8), 0.6, hii, 0.4, 0)

                        to_show.append(np.asarray(hii, dtype=np.float32) / 255.0)

                    to_show = torch.from_numpy(np.asarray(to_show, dtype=np.float32).transpose(0, 3, 1, 2))
                    grid = tv.utils.make_grid(to_show, normalize=True, range=(0, 1), nrow=x.shape[0])
                    grid = (grid.cpu().numpy() * 255).astype(np.uint8).transpose(1,2,0)
                    tf.summary.image(name=f'results_{step}', data=grid, step=self.epoch)
                    self.sw.flush()

            # log average loss on test set
            mean_test_mse = (self.test_h_mse_metric.result() + self.test_cnt_mse_metric.result() + self.test_s_mse_metric.result()) / 3
            print(f'\t● AVG MSE on TEST-set: {mean_test_mse:.6f} │ T: {time() - t:.2f} s')

            tf.summary.scalar(name='test_h_mse', data=self.test_h_mse_metric.result(), step=self.epoch)
            tf.summary.scalar(name='test_cnt_mse', data=self.test_cnt_mse_metric.result(), step=self.epoch)
            tf.summary.scalar(name='test_s_mse', data=self.test_s_mse_metric.result(), step=self.epoch)
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