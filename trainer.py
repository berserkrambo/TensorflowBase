# -*- coding: utf-8 -*-
# ---------------------


from conf import Conf
from dataset.widerface import DataGenerator
from models.model import get_model

import tensorflow as tf
from tensorflow import keras
from progress_bar import ProgressBar
from time import time
import numpy as np
import cv2
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from dataset.dataset import Data


class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> Trainer
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.cnf = cnf

        # init train loader
        # self.train_loader = DataGenerator(self.cnf, partition='train', shuffle=True)
        # self.test_loader = DataGenerator(self.cnf, partition='test', shuffle=False)

        # init train loader
        def asd(worker_id):
            return np.random.seed(worker_id)

        training_set = Data(cnf, partition='train')
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=True, pin_memory=True, worker_init_fn=asd, drop_last=True
        )

        # init test loader
        test_set = Data(cnf, partition='test')
        self.test_loader = DataLoader(
            dataset=test_set, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False, drop_last=True
        )

        # init model
        self.model = get_model(input_shape=self.cnf.input_shape, model_str=self.cnf.model)

        # init optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.cnf.lr)

        # init loss
        self.loss_mse = tf.keras.losses.MeanSquaredError()
        # self.loss_mae = tf.keras.losses.MeanAbsoluteError()

        # init metrics
        self.test_h_mse_metric = keras.metrics.MeanSquaredError()
        # self.test_cnt_mse_metric = keras.metrics.MeanSquaredError()
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

        # starting values
        self.epoch = 0
        self.best_test_loss = None

        # init progress bar
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

        # possibly load checkpoint
        self.load_ck()
        self.reset_metric()

    def reset_metric(self):
        self.train_losses = {"mse_c": [], "mse_s": [], "mse_cnt": [], "total": []}
        self.test_losses = {"mse_c": [], "mse_s": [], "mse_cnt": [], "total": []}

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
        stop = self.train_loader.__len__() - 1

        for step, sample in enumerate(self.train_loader):
            t = time()

            x, y_true, meta = sample
            x = x.numpy().transpose(0,2,3,1)
            # meta = meta.numpy().transpose(0,2,3,1)
            y_true_h, y_true_sz = y_true[0].numpy().transpose(0,2,3,1), y_true[1].unsqueeze(3).numpy()
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                y_pred_h, y_pred_s = self.model(x, training=True)

                # Compute the loss value for this minibatch.
                loss_h = self.loss_mse(y_true_h, y_pred_h)
                # loss_cnt = self.loss_mse(y_true_cnt, y_pred_cnt)
                loss_s = self.loss_mse(y_true_sz, y_pred_s)

                loss = loss_h + loss_s

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss, self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            self.train_losses["mse_c"].append(loss_h)
            self.train_losses["mse_s"].append(loss_s)
            # self.train_losses["mse_cnt"].append(loss_cnt)
            self.train_losses["total"].append(loss)

            if step == stop:
                grid = self.get_grid_view(x,y_pred_h, y_pred_s)
                with self.sw.as_default():
                    tf.summary.image(name=f'results_train', data=grid, step=self.epoch)
                    self.sw.flush()

            # print an incredible progress bar
            times.append(time() - t)
            if self.cnf.log_each_step or (not self.cnf.log_each_step and self.progress_bar.progress == 1):
                print(f'\r{self.progress_bar} '
                      f'│ Loss: {np.mean(self.train_losses["total"]):.6f} '
                      f'│ ↯: {1 / np.mean(times):5.2f} step/s', end='')
            self.progress_bar.inc()

        # log average loss of this epoch
        with self.sw.as_default():
            tf.summary.scalar(name='train_loss', data=np.mean(self.train_losses["total"]), step=self.epoch)
            # tf.summary.scalar(name='train_mse_c', data=np.mean(self.train_losses["mse_c"]), step=self.epoch)
            tf.summary.scalar(name='train_mse_s', data=np.mean(self.train_losses["mse_s"]), step=self.epoch)
            tf.summary.scalar(name='train_mse_cnt', data=np.mean(self.train_losses["mse_cnt"]), step=self.epoch)
            self.sw.flush()
        self.reset_metric()

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')

    def test(self):
        """
        test model on the Test-Set
        """

        t = time()
        with self.sw.as_default():
            stop = self.test_loader.__len__() - 1

            for step, sample in enumerate(self.test_loader):
                x, y_true, meta = sample
                x = x.numpy().transpose(0, 2, 3, 1)
                # meta = meta.numpy().transpose(0,2,3,1)
                y_true_h, y_true_sz = y_true[0].numpy().transpose(0, 2, 3, 1), y_true[1].unsqueeze(3).numpy()
                y_pred_h, y_pred_s = self.model(x, training=False)

                self.test_h_mse_metric.update_state(y_true_h, y_pred_h)
                # self.test_cnt_mse_metric.update_state(y_true_cnt, y_pred_cnt)
                self.test_s_mse_metric.update_state(y_true_sz, y_pred_s)

                # draw results for this step in a 3 rows grid:
                # row #1: input (x)
                # row #2: predicted_output (y_pred)
                # row #3: target (y_true)

                if step == stop:
                    grid = self.get_grid_view(x,y_pred_h, y_pred_s)
                    with self.sw.as_default():
                        tf.summary.image(name=f'results_test', data=grid, step=self.epoch)
                        self.sw.flush()

            # log average loss on test set
            mean_test_mse = self.test_h_mse_metric.result() + self.test_s_mse_metric.result()
            print(f'\t● AVG MSE on TEST-set: {mean_test_mse:.6f} │ T: {time() - t:.2f} s')

            with self.sw.as_default():
                tf.summary.scalar(name='test_h_mse', data=self.test_h_mse_metric.result(), step=self.epoch)
                # tf.summary.scalar(name='test_cnt_mse', data=self.test_cnt_mse_metric.result(), step=self.epoch)
                tf.summary.scalar(name='test_s_mse', data=self.test_s_mse_metric.result(), step=self.epoch)
                tf.summary.scalar(name='test_loss', data=mean_test_mse, step=self.epoch)
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
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quant 8 bit

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

            with tf.io.gfile.GFile(self.cnf.tflite_model_outpath / 'model.tflite', 'wb') as f:
                f.write(tflite_model)

    def get_grid_view(self, x, hm, sz):
        x_np = (x * 255).astype(np.uint8)
        hm, sz = hm.numpy(), sz.numpy().squeeze()
        hm = torch.from_numpy(hm.transpose(0, 3, 2, 1))
        to_show = []

        # simple nms
        hmax = torch.nn.functional.max_pool2d(hm, kernel_size=5, padding=2, stride=1)
        keep = (hmax == hm).float()
        hm *= keep
        hm = hm.numpy()
        for bi in range(len(x_np)):
            hm_np = hm[bi]
            sz_np = sz[bi]

            xi = x_np[bi].copy()
            for chi in range(0, self.cnf.ds_classify + 1):
                ind = np.argpartition(hm_np[chi].squeeze().flatten(), -88)[-88:]

                # out = np.zeros(shape=(self.cnf.input_shape[0], self.cnf.input_shape[1], 3 ), dtype=np.uint8)
                for v in ind:
                    row = v % hm_np[chi].shape[0]
                    col = v // hm_np[chi].shape[0]

                    if hm_np[chi][col, row] <= 0.1:
                        continue
                    szi = int(sz_np[col, row] * self.cnf.input_shape[0])

                    row *= self.cnf.stride
                    col *= self.cnf.stride
                    szi = szi // 2
                    x1, y1, x2, y2 = row - szi, col - szi, row + szi, col + szi
                    if x1 < 0:
                        x1 = 0
                    if y1 < 0:
                        y1 = 0
                    if x2 >= self.cnf.input_shape[0]:
                        x2 = self.cnf.input_shape[0] - 1
                    if y2 >= self.cnf.input_shape[1]:
                        y2 = self.cnf.input_shape[1] - 1

                    cv2.rectangle(xi, (x1, y1), (x2, y2), (255*(1-chi), 0, 255*chi), 2)
                # out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
                # out = cv2.addWeighted((x_np[bi] * 255).astype(np.uint8), 0.6, out, 0.4, 0)

            to_show.append(np.asarray(xi, dtype=np.float32) / 255.0)
        to_show = torch.from_numpy(np.asarray(to_show, dtype=np.float32).transpose(0, 3, 1, 2))
        grid = tv.utils.make_grid(to_show, normalize=True, range=(0, 1), nrow=x.shape[0])
        grid = (grid.cpu().numpy() * 255).astype(np.uint8).transpose(1,2,0)
        return grid

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
