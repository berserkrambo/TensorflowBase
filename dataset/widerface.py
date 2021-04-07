# -*- coding: utf-8 -*-
# ---------------------


import numpy as np
from tensorflow import keras
from utils import imread_cv
from dataset.utils import letterbox, draw_gaussian, gaussian_radius
import xml.etree.ElementTree as ET
import cv2
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, cnf, shuffle=True, partition='train'):
        'Initialization'
        self.cnf = cnf
        self.partition = partition
        assert self.partition in ['test', 'train', 'val']

        self.labels = []
        if self.partition == "train":
            txt_path = self.cnf.ds_path / "WIDER_train" / "wider_face_train_bbx_gt.txt"
        elif self.partition == "test":
            txt_path = self.cnf.ds_path / "WIDER_val" / "wider_face_val_bbx_gt.txt"

        self.img_path = []
        self.data = {}
        f = open(txt_path, 'r')
        lines = f.readlines()
        f.close()

        i = 0
        while i < lines.__len__():
            actual_path = txt_path.parent / "images" / lines[i].strip()
            actual_labels = []
            i += 1
            num_anno = int(lines[i])
            if num_anno > 0:
                for na in range(num_anno):
                    i += 1
                    x1, y1, w, h = [int(b) for b in lines[i].strip().split()[:4]]
                    if w <= 0 or h <= 0:
                        continue
                    x2, y2 = x1 + w, y1 + h
                    actual_labels.append(np.asarray([x1, y1, x2, y2]))
                if actual_labels.__len__() > 0:
                    self.data[actual_path] = np.asarray(actual_labels)
                    self.img_path.append(actual_path)
            else:
                i += 1
            i += 1

        self.seq = iaa.SomeOf((0, 3), [
            iaa.GaussianBlur((0.25, 0.3)),
            iaa.AverageBlur(k=(1, 15)),
            iaa.SaltAndPepper(p=0.01),
            iaa.LinearContrast((0.5, 1.5)),
            iaa.GammaContrast(per_channel=True),
            iaa.Grayscale(),
        ])
        self.seq_affine = iaa.Sequential([
            iaa.Fliplr(),
            iaa.Rotate(rotate=(-10, 10))
        ])

        self.len = len(self.data)
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
        list_IDs_temp = [self.img_path[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)


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


        # Generate data
        for i, path in enumerate(list_IDs_temp):

            annos = self.data[path]
            s = self.cnf.stride

            img_input = imread_cv(path)

            if self.partition == 'train':
                img_input = self.seq.augment_image(img_input)

            img_input, ratio, ds = letterbox(img_input.copy(), new_shape=self.cnf.input_shape[0], color=(0, 0, 0),
                                             auto=False)

            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=((x1 * ratio[0]) + ds[0]),
                            y1=((y1 * ratio[1]) + ds[1]),
                            x2=((x2 * ratio[0]) + ds[0]),
                            y2=((y2 * ratio[1]) + ds[1])) for x1, y1, x2, y2 in annos
            ], shape=img_input.shape)

            if self.partition == 'train':
                # img_input = self.seq.augment_image(img_input)
                affine = self.seq_affine.to_deterministic()
                img_input, bbs = affine(image=img_input, bounding_boxes=bbs)
                bbs = bbs.remove_out_of_image().clip_out_of_image()

            bbs_s = BoundingBoxesOnImage([
                BoundingBox(x1=bb.x1 / s, y1=bb.y1 / s, x2=bb.x2 / s, y2=bb.y2 / s) for bb in bbs.bounding_boxes
            ], shape=tuple([sz // s for sz in img_input.shape]))
            bbs_s = bbs_s.remove_out_of_image().clip_out_of_image()

            hm = np.zeros(shape=(self.cnf.input_shape[1] // s, self.cnf.input_shape[0] // s), dtype=np.float32)
            sz = np.zeros(shape=(self.cnf.input_shape[1] // s, self.cnf.input_shape[0] // s), dtype=np.float32)
            cnts = np.zeros(shape=(2, self.cnf.input_shape[1] // s, self.cnf.input_shape[0] // s), dtype=np.float32)

            bboxes = []
            for bbox in bbs.bounding_boxes:
                bboxes.append([bbox.x1_int, bbox.y1_int, bbox.x2_int, bbox.y2_int])
                # cnt = int(bbox.center_x), int(bbox.center_y)
                # cv2.circle(img_input, (cnt[0], cnt[1]), 4, [0, 0, 255])

            for bbox in bbs_s.bounding_boxes:
                cnt_hm = [int(bbox.center_x), int(bbox.center_y)]

                sz[bbox.y1_int:bbox.y2_int, bbox.x1_int:bbox.x2_int] = max(bbox.height, bbox.width) / (
                            self.cnf.input_shape[0] // s)
                cnts[0, bbox.y1_int:bbox.y2_int, bbox.x1_int:bbox.x2_int] = bbox.center_x / (
                            self.cnf.input_shape[0] // s)
                cnts[1, bbox.y1_int:bbox.y2_int, bbox.x1_int:bbox.x2_int] = bbox.center_x / (
                            self.cnf.input_shape[1] // s)

                r = gaussian_radius([int(bbox.height), int(bbox.width)])
                hm[cnt_hm[1], cnt_hm[0]] = 1.0
                draw_gaussian(hm, cnt_hm, r)
                hm[cnt_hm[1], cnt_hm[0]] = 1.0

            # cv2.imshow("img_input", img_input)
            # cv2.imshow("hm", cv2.resize(hm, (self.cnf.input_shape[0], self.cnf.input_shape[1])))
            # cv2.imshow("sz", sz)
            # cv2.waitKey()

            bboxes = np.asarray(bboxes)

            img_input = (img_input / 255.0).astype(np.float32).transpose(2, 0, 1)

            X.append(img_input)
            heatmaps.append(hm)
            sizes.append(sz)
            heatmaps_centers.append(cnts)

        X = np.asarray(X).transpose(0,2,3,1)

        heatmaps = np.expand_dims(np.asarray(heatmaps), axis=3)
        sizes = np.expand_dims(np.asarray(sizes), axis=3)
        heatmaps_centers = np.asarray(heatmaps_centers).transpose(0, 2, 3, 1)

        return X, [heatmaps, heatmaps_centers, sizes]