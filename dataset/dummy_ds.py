# -*- coding: utf-8 -*-
# ---------------------


import numpy as np
from tensorflow import keras
from utils import imread_cv
from utils import letterbox


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, cnf, shuffle=True, partition='train'):
        'Initialization'
        self.cnf = cnf
        self.partition = partition
        assert self.partition in ['test', 'train', 'val']

        self.labels = []
        # read labels file
        with open(self.cnf.ds_path / 'labels.txt') as f:
            for i,l in enumerate(f.readlines()):
                self.labels.append([i,l.strip()])

        # read imgs list and set tuple img,lbl
        self.data = []
        for l in self.labels:
            ls = sorted((self.cnf.ds_path / self.partition / l[1]).files('*.jpg'))
            for i in ls:
                self.data.append([i, l[0]])

        self.n_classes = len(self.labels)
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.cnf.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.cnf.batch_size:(index+1)*self.cnf.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.data[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        # if self.partition in ['train', 'val']:
        #     return X, y
        # elif self.partition == 'test':
        #     return X

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = imread_cv(ID[0])
            img, ratio, ds = letterbox(img,new_shape=self.cnf.input_shape[:2], color=(0,0,0), auto=False)
            X.append((img / 2550))

            # Store class
            y.append(ID[1])

        # bo vabe qua tf vuole una matrice anzichè un vettore anzichè uno scalare....booooo
        return np.asarray(X), keras.utils.to_categorical(y,num_classes=self.n_classes)


def main():
    ds = DataGenerator()

    for i in range(10):
        x, y = ds[i]
        print(f'Example #{i}: x.shape={x.shape}, y.shape={y.shape}')


if __name__ == '__main__':
    main()
