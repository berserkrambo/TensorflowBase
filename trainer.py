# -*- coding: utf-8 -*-
# ---------------------

from time import time

import numpy as np


from conf import Conf
from dataset.dummy_ds import DataGenerator
from models.model import DummyModel
from progress_bar import ProgressBar

import tensorflow as tf
from tensorflow import keras
import tensorboard

class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        self.cnf = cnf

        # init train loader
        self.train_loader = DataGenerator(self.cnf)
        self.test_loader = DataGenerator(self.cnf, partition='test', shuffle=False)

        # init model
        self.model = DummyModel(self.cnf.input_shape)

        # init optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.cnf.lr)

        # init loss
        self.loss = tf.keras.losses.CategoricalCrossentropy()

        # compile the model
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        print(f'tensorboard --logdir={cnf.project_log_path.abspath()} --samples_per_plugin=4096\n')
        # self.sw = tf.summary.SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)
        self.train_losses = []
        self.test_losses = []

        # starting values
        self.epoch = 0
        self.best_test_loss = None

        # possibly load checkpoint
        self.load_ck()


    def load_ck(self):
        """
        load training checkpoint
        """
        pass


    def save_ck(self):
        """
        save training checkpoint
        """
        pass


    def train(self):
        """
        train model for one epoch on the Training-Set.
        """

        # fit the model
        self.model.fit(self.train_loader, epochs=1)


    def test(self):
        """
        test model on the Test-Set
        """

        loss, acc = self.model.evaluate(self.test_loader)

        self.test_losses = []
        # self.sw.add_scalar(tag='test_loss', scalar_value=loss, global_step=self.epoch)

        # save best model
        if self.best_test_loss is None or loss < self.best_test_loss:
            self.best_test_loss = loss

    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for _ in range(self.epoch, self.cnf.epochs):
            self.train()

            self.test()

            self.epoch += 1
            self.save_ck()
