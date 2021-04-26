# -*- coding: utf-8 -*-
# ---------------------


import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from utils import imread_cv
from dataset.utils import letterbox, draw_gaussian, gaussian_radius
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import xml.etree.ElementTree as ET
import cv2
import dlib
from scipy.spatial import Delaunay
from multiprocessing.pool import Pool
import pickle


def dlib_on_widerface(dlib_path, txt_path, out_path):
    nproc = 6
    pool = Pool(processes=nproc)
    tmp_data, tmp_paths = parse_widerface(txt_path)
    _idx = np.linspace(0, len(tmp_paths), nproc + 1, dtype=np.int)

    results = [pool.apply_async(run_dlib, (dlib_path, tmp_paths[_idx[pr]:_idx[pr+1]],)) for pr in range(nproc)]
    final_res = {}
    for resi, res in enumerate(results):
        final_res.update(res.get())
    pool.close()
    pool.join()

    with open(out_path, 'wb') as f:
        pickle.dump(final_res, f)

def run_dlib(dlib_path, paths):
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(dlib_path / "shape_predictor_68_face_landmarks.dat")
    out = {}
    for path in paths:
        image = imread_cv(path)
        faces = face_detector(image, 1)
        landmarks = []
        for k, d in enumerate(faces):
            landmarks.append(landmark_detector(image, d))
        out[path] = [faces, landmarks]
    return out

def parse_widerface(txt_path):
    data = {}
    img_path = []

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
                actual_labels.append(np.asarray([x1, y1, x2, y2, 0]))  # x1, y1, x2, y2, 0/1 -> nomask/mask
            if actual_labels.__len__() > 0:
                data[actual_path] = np.asarray(actual_labels)
                img_path.append(actual_path)
        else:
            i += 1
        i += 1

    return data, img_path


class Data(Dataset):
    def __init__(self, cnf, partition='train'):
        'Initialization'
        self.cnf = cnf
        self.partition = partition
        assert self.partition in ['test', 'train', 'val']

        # assert not (self.cnf.ds_classify and 'widerface' in self.cnf.ds_path), "widerface and classify cannot coexists"

        self.labels = []
        self.img_path = []
        self.data = {}
        self.dlib_data = {}
        parent_path = ""

        for pk, pv in self.cnf.ds_path.items():
            parent_path = pv.parent
            if pk == "widerface":

                if self.partition == "train":
                    txt_path = pv / "WIDER_train" / "wider_face_train_bbx_gt.txt"
                elif self.partition == "test":
                    txt_path = pv / "WIDER_val" / "wider_face_val_bbx_gt.txt"

                data, tmp_img_path = parse_widerface(txt_path=txt_path)
                img_path = []
                for path in tmp_img_path:
                    img_path.append([path,"widerface"])
                self.img_path = self.img_path + img_path
                self.data.update(data)

                if self.cnf.ds_classify:
                    dlib_file = f"dlib_data_{self.partition}.pkl"
                    if not (pv / dlib_file).exists():
                        print(f"processing dlib_data_{self.partition}")
                        dlib_on_widerface(dlib_path=parent_path/"dlib", txt_path=txt_path, out_path=pv / dlib_file)
                    with open(pv / dlib_file, "rb") as pf:
                        print(f"opening dlib_data_{self.partition}")
                        self.dlib_data = pickle.load(pf)

            elif pk == "face_mask":
                # read annotation file
                annotation_dir = pv / self.partition
                labels = sorted(annotation_dir.files('*.xml'))

                # read imgs list and set tuple img,lbl
                data = [l.replace('xml', 'png') for l in labels]

                for item in data:
                    label_file = ET.parse(item.replace("png", "xml"))
                    label_root = label_file.getroot()
                    bboxes = []
                    # for bi, bb in enumerate(label_root.iter('bndbox')):
                    #     xmin = int(bb.find('xmin').text)
                    #     ymin = int(bb.find('ymin').text)
                    #     xmax = int(bb.find('xmax').text)
                    #     ymax = int(bb.find('ymax').text)
                    #     bboxes.append(np.asarray([xmin, ymin, xmax, ymax])) # x1, y1, x2, y2, 0/1 -> nomask/mask
                    for oi, oo in enumerate(label_root.iter('object')):
                        mask = 0 if oo.find("name").text.lower() == "without_mask" else 1
                        bb = oo.find('bndbox')
                        xmin = int(bb.find('xmin').text)
                        ymin = int(bb.find('ymin').text)
                        xmax = int(bb.find('xmax').text)
                        ymax = int(bb.find('ymax').text)
                        bboxes.append(np.asarray([xmin, ymin, xmax, ymax, mask])) # x1, y1, x2, y2, 0/1 -> nomask/mask
                    self.data[item] = np.asarray(bboxes)
                    self.img_path.append([item, "face_mask"])

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

        self.masks = []
        for mp in sorted((parent_path / "masks").files("*.png")):
            im_mask = cv2.imread(mp, -1)
            im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGRA2RGBA)
            points = []
            with open(mp.replace(".png", ".txt")) as fm:
                lines = fm.readlines()
                for line in lines:
                    x,y = [int(c) for c in line.strip().split(",")]
                    points.append(np.asarray([x,y]))
            points = np.asarray(points, dtype=np.float32)
            self.masks.append([im_mask, points])

    def __len__(self):
        # type: () -> int
        return self.len

    def __getitem__(self, i):
        # type: (int) -> Tuple[Tensor, Tensor]

        path, im_ds_type = self.img_path[i]
        annos = self.data[path]
        s = self.cnf.stride
        meta = {}

        img_input = imread_cv(path)

        if self.cnf.ds_classify and im_ds_type == "widerface":
            img_input_rgba = cv2.cvtColor(img_input, cv2.COLOR_RGB2RGBA)

            faces = self.dlib_data[path][0]
            to_drop = -1
            if len(faces) > 1:
                to_drop = np.random.randint(len(faces) - 1)

            for k, d in enumerate(faces):
                if to_drop >= 0 and k == to_drop:
                    continue
                landmark_tuple = []
                landmarks = self.dlib_data[path][1][k]
                for n in [1, 15, 5, 11]:
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmark_tuple.append(np.asarray([x,y]))
                    # cv2.circle(img_input, (x, y), 2, (255, 255, 0), -1)

                landmark_tuple = np.asarray(landmark_tuple, dtype=np.float32)
                n_mask = np.random.randint(len(self.masks))
                h = cv2.getPerspectiveTransform(self.masks[n_mask][1], landmark_tuple)
                temp = cv2.warpPerspective(self.masks[n_mask][0], h, (img_input.shape[1], img_input.shape[0]))

                img_input_rgba[temp[:,:, 3] != 0] = temp[temp[:,:, 3] != 0]
                img_input = cv2.cvtColor(img_input_rgba, cv2.COLOR_RGBA2RGB)

                for ai in range(len(annos)):
                    x0,y0,x2,y2 = annos[ai][:4]
                    hull = np.asarray([[x0,y0],[x2,y0],[x2,y2],[x0,y2]])
                    if not isinstance(hull, Delaunay):
                        hull = Delaunay(hull)
                    p = []
                    p.append([landmarks.part(33).x, landmarks.part(33).y])
                    p = np.asarray(p, dtype=np.float32)
                    if hull.find_simplex(p) >= 0:
                        annos[ai][4] = 1
                        # cv2.circle(img_input, (landmarks.part(33).x, landmarks.part(33).y), 2, (255, 255, 0), -1)
                        # for n in [1, 15, 5, 11]:
                        #     x = landmarks.part(n).x
                        #     y = landmarks.part(n).y
                        #     cv2.circle(img_input, (x, y), 2, (255, 255, 0), -1)
                # cv2.imshow("", img_input)
                # cv2.waitKey()

        if self.partition == 'train':
            img_input = self.seq.augment_image(img_input)

        img_input, ratio, ds = letterbox(img_input.copy(), new_shape=self.cnf.input_shape[0], color=(0, 0, 0), auto=False, scaleup=False)
        meta["ratio"] = ratio
        meta["ds"] = ds

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=((x1 * ratio[0]) + ds[0]),
                        y1=((y1 * ratio[1]) + ds[1]),
                        x2=((x2 * ratio[0]) + ds[0]),
                        y2=((y2 * ratio[1]) + ds[1]), label=mask) for x1, y1, x2, y2, mask in annos
        ], shape=img_input.shape)

        if self.partition == 'train':
            # img_input = self.seq.augment_image(img_input)
            affine = self.seq_affine.to_deterministic()
            img_input, bbs = affine(image=img_input, bounding_boxes=bbs)
            bbs = bbs.remove_out_of_image().clip_out_of_image()

        bbs_s = BoundingBoxesOnImage([
            BoundingBox(x1=bb.x1 / s, y1=bb.y1 / s, x2=bb.x2 / s, y2=bb.y2 / s, label=bb.label) for bb in bbs.bounding_boxes
        ], shape=tuple([sz // s for sz in img_input.shape]))
        bbs_s = bbs_s.remove_out_of_image().clip_out_of_image()

        hm_shape = 1
        if self.cnf.ds_classify:
            hm_shape = 2
        hm = np.zeros(shape=(hm_shape, self.cnf.input_shape[1] // s, self.cnf.input_shape[0] // s), dtype=np.float32)
        sz = np.zeros(shape=(self.cnf.input_shape[1] // s, self.cnf.input_shape[0] // s), dtype=np.float32)
        # cnts = np.zeros(shape=(2, self.cnf.input_shape[1] // s, self.cnf.input_shape[0] // s), dtype=np.float32)

        # bboxes = []
        # for bbox in bbs.bounding_boxes:
        #     bboxes.append([bbox.x1_int, bbox.y1_int, bbox.x2_int, bbox.y2_int])
            # cnt = int(bbox.center_x), int(bbox.center_y)
            # cv2.circle(img_input, (cnt[0], cnt[1]), 4, [0, 0, 255])

        for bbox in bbs_s.bounding_boxes:
            cnt_hm = [int(bbox.center_x), int(bbox.center_y)]

            sz[bbox.y1_int:bbox.y2_int, bbox.x1_int:bbox.x2_int] = max(bbox.height, bbox.width) / (self.cnf.input_shape[0] // s)
            # cnts[0, bbox.y1_int:bbox.y2_int, bbox.x1_int:bbox.x2_int] = bbox.center_x / (self.cnf.input_shape[0] // s)
            # cnts[1, bbox.y1_int:bbox.y2_int, bbox.x1_int:bbox.x2_int] = bbox.center_y / (self.cnf.input_shape[1] // s)

            r = gaussian_radius([int(bbox.height), int(bbox.width)])

            ch = 0
            if self.cnf.ds_classify:
                ch = int(bbox.label)
            hm[ch, cnt_hm[1], cnt_hm[0]] = 1.0
            draw_gaussian(hm[ch], cnt_hm, r)
            hm[ch, cnt_hm[1], cnt_hm[0]] = 1.0

        # cv2.imshow("img_input", img_input)
        # cv2.imshow("hm", cv2.resize(hm, (self.cnf.input_shape[0], self.cnf.input_shape[1])))
        # cv2.imshow("sz", sz)
        # cv2.waitKey()

        hm = hm.astype(np.float32)
        sz = sz.astype(np.float32)
        # cnts = cnts.astype(np.float32)
        # bboxes = np.asarray(bboxes)

        img_input = (img_input / 255.0).astype(np.float32).transpose(2, 0, 1)

        return img_input, [hm, sz], meta
