# -*- coding: utf-8 -*-
# ---------------------


import numpy as np
from tensorflow import keras
from utils import imread_cv
from dataset.utils import letterbox, draw_gaussian, gaussian_radius
import xml.etree.ElementTree as ET
import cv2


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, cnf, shuffle=True, partition='train'):
        'Initialization'
        self.cnf = cnf
        self.partition = partition
        assert self.partition in ['test', 'train', 'val']

        self.labels = []
        # read annotation file
        annotation_dir = self.cnf.ds_path / self.partition
        self.labels = sorted(annotation_dir.files('*.xml'))

        # read imgs list and set tuple img,lbl
        self.data = [l.replace('xml', 'png') for l in self.labels]

        self.n_classes = 2
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.cnf.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.cnf.batch_size:(index + 1) * self.cnf.batch_size]

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
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        sizes = []
        heatmaps = []
        heatmaps_centers = []

        # fix coords
        # augment image -> # fix coords
        # generate gts

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = imread_cv(ID)
            # img_lt, ratio, ds = letterbox(img,new_shape=self.cnf.input_shape[:2], color=(0,0,0), auto=False)
            # X.append((img / 255.0).astype(np.float32))
            cnts = np.zeros(shape=(1,64,64))
            hm = np.zeros(shape=(img.shape[0], img.shape[1]))

            label = ET.parse(ID.replace("png", "xml"))
            label_root = label.getroot()
            ymin, ymax, xmin, xmax = 0, 0, 0, 0
            for bi, bbox in enumerate(label_root.iter('bndbox')):
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                cnt = ((xmin + xmax) // 2, (ymin + ymax) // 2)

                r = gaussian_radius((int(ymax-ymin), int(xmax-xmin)))
                draw_gaussian(hm, cnt, r)
                cnts[bi] = cnt
                # cv2.circle(img,cnt,3,[0,0,255],1)
                # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [0, 0, 255], 1)
                # cv2.imshow(f"{i}_{bi}.jpg", img[ymin:ymax,xmin:xmax])
            # cv2.imshow(f"{i}", img)
            # cv2.imshow(f"{i}_h", (hm * 255.).astype(np.uint8))
            # cv2.waitKey()

            X.append(img)
            heatmaps.append(hm)

        return np.asarray(X), keras.utils.to_categorical(y, num_classes=self.n_classes)


def main():
    ds = DataGenerator()

    for i in range(10):
        x, y = ds[i]
        print(f'Example #{i}: x.shape={x.shape}, y.shape={y.shape}')


if __name__ == '__main__':
    main()
