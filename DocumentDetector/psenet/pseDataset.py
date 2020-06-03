# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import os

random.seed(123456)


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs


def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def random_scale(img, min_size):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[0:2]
    random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis=1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)

        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs


def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        offset = int(area * (1 - rate) / (peri + 0.001) + 0.5)

        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bbox = np.array(shrinked_bbox)[0]
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bboxes.append(shrinked_bbox)

    return np.array(shrinked_bboxes)


def my_shrink(bboxes: np.ndarray, rate):
    pass


def scale_image_with_gt(img: np.array, gt, size):
    height = img.shape[0]
    width = img.shape[1]
    img = cv2.resize(img, size)
    factor_x = img.shape[1] / width
    factor_y = img.shape[0] / height
    scaled_gt = []
    for box in gt:
        scaled_box = []
        for i in range(len(box)):
            if (i % 2) == 0:
                scaled_box.append(int(box[i] * factor_x))
            else:
                scaled_box.append(int(box[i] * factor_y))
        scaled_gt.append(scaled_box)
    return img, scaled_gt


class IC15Loader(data.Dataset):
    def __init__(self, img_dir, label_dir, is_transform=False, img_size=None, kernel_num=7, min_scale=0.4):
        self.is_transform = is_transform
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_num = kernel_num
        self.min_scale = min_scale
        self.img_dir = img_dir
        self.labels = []
        self.img_list = []
        for sdir in img_dir:
            files = os.listdir(sdir)
            for file in files:
                if file[-3:] != 'jpg' and file[-3:] != 'png':
                    continue
                self.img_list.append(os.path.join(sdir, file))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        gt_path = img_path[:-3] + 'txt'
        bboxes = []
        fp = open(gt_path, 'r', encoding='utf-8')
        for line in fp.readlines():
            pt = line.strip('\n').split(',')
            box = [int(pt[i]) for i in range(8)]
            bboxes.append(box)
        fp.close()
        img = cv2.imread(img_path)
        img, scaled_boxes = scale_image_with_gt(img, bboxes, (800, 800))
        bboxes = np.array(scaled_boxes)
        bboxes = bboxes.reshape(bboxes.shape[0], 4, 2)
        img = img[:, :, [2, 1, 0]]

        if self.is_transform:
            img = random_scale(img, self.img_size[0])

        gt_text = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)

        # cv2.imshow("img", gt_text)
        # cv2.waitKey(0)

        gt_kernels = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        # pp = np.zeros(gt_text.shape[0:2], dtype=np.uint8)
        # pp[gt_text > 0] = 255
        # for idx in range(len(gt_kernels)):
        #     pp[gt_kernels[idx] > 0] = 255 - (255 - 20) / (len(gt_kernels) - 1) * (idx + 1)
        # cv2.imshow("pp" + img_path, pp)
        # cv2.imshow("ori img"+img_path, img)
        # cv2.waitKey(0)

        if self.is_transform:
            imgs = [img, gt_text, training_mask]
            imgs.extend(gt_kernels)

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop(imgs, self.img_size)

            img, gt_text, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]

        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        # '''
        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).float()
        gt_kernels = torch.from_numpy(gt_kernels).float()
        training_mask = torch.from_numpy(training_mask).float()
        # '''

        return img, gt_text, gt_kernels, training_mask


if __name__ == '__main__':
    print("running dataLoader test")
    bbox = np.array([[[633, 343],
                      [227, 364],
                      [214, 121],
                      [620, 100]],

                     [[515, 719],
                      [106, 733],
                      [98, 488],
                      [507, 475]]])
    kernel_box = shrink(bbox, 0.4)
    print("rate: 0.4")
    print(kernel_box)
    kernel_box = shrink(bbox, 0.9)
    print("rate: 0.2")
    print(kernel_box)
