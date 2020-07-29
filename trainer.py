# -*- coding: utf-8 -*-
# ---------------------


from conf import Conf
from dataset.dummy_ds import DataGenerator
from models.model import get_DummyModel

import tensorflow as tf
from tensorflow import keras


class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        self.cnf = cnf

        # init train loader
        self.train_loader = DataGenerator(self.cnf)
        self.val_loader = DataGenerator(self.cnf, partition='val', shuffle=False)
        self.test_loader = DataGenerator(self.cnf, partition='test', shuffle=False)

        # init model
        self.model = get_DummyModel(input_shape=self.cnf.input_shape)

        # init optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.cnf.lr)

        # init loss
        self.loss = tf.keras.losses.BinaryCrossentropy()

        # compile the model
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        # init logging stuffs

        self.callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.cnf.exp_log_path),
            keras.callbacks.ModelCheckpoint(self.cnf.exp_weights_path, verbose=1, save_best_only=True),
            keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
        ]

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

            # Convert the model.
            converter = tf.lite.TFLiteConverter.from_saved_model(self.cnf.exp_weights_path)
            tflite_model = converter.convert()

            with tf.io.gfile.GFile('model.tflite', 'wb') as f:
                f.write(tflite_model)

    def export_tflite(self):
        """
        convert saved checkpoint to tflite
        """

        if self.cnf.exp_weights_path.exists():

            # Convert the model.
            converter = tf.lite.TFLiteConverter.from_saved_model(self.cnf.exp_weights_path)
            tflite_model = converter.convert()

            self.cnf.tflite_model_outpath.makedirs_p()

            with tf.io.gfile.GFile(self.cnf.tflite_model_outpath /'model.tflite', 'wb') as f:
                f.write(tflite_model)

    def train(self):
        """
        train model for one epoch on the Training-Set.
        """

        # fit the model
        self.model.fit(self.train_loader, epochs=self.cnf.epochs, validation_data=self.val_loader,
                       callbacks=self.callbacks, use_multiprocessing=True, workers=self.cnf.n_workers)

    def test(self):
        """
        test model on the Test-Set
        """

        loss, acc = self.model.evaluate(self.test_loader)
        print("accuracy on test", acc)

    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        self.train()

        self.test()

        if self.cnf.export_tflite:
            self.export_tflite()