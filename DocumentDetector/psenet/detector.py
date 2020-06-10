import torch
import cv2
import pseNets
import numpy as np
from PIL import Image
from torchvision import transforms
from pypse import pse as pypse
import collections


class Detector:
    def __init__(self):
        self.model = pseNets.resnet50(pretrained=True, num_classes=7)  # type:torch.nn.Module
        for param in self.model.parameters():
            param.requires_grad = False

    def load_weights(self, model_path: str):
        checkpoint = torch.load(model_path, map_location='cpu')
        d = collections.OrderedDict()
        for key, value in checkpoint['state_dict'].items():
            tmp = key[7:]
            d[tmp] = value
        self.model.load_state_dict(d)
        self.model.eval()

    def predict(self, img_path):
        org_img = cv2.imread(img_path)
        img = org_img[:, :, [2, 1, 0]]
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        img = img.unsqueeze(0)

        outputs = self.model(img)

        score = torch.sigmoid(outputs[:, 0, :, :])
        outputs = (torch.sign(outputs - 1.0) + 1) / 2

        text = outputs[:, 0, :, :]
        kernels = outputs[:, 0:7, :, :] * text

        score = score.data.cpu().numpy()[0].astype(np.float32)
        text = text.data.cpu().numpy()[0].astype(np.uint8)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

        pred = pypse(kernels, 5.0 / (1 * 1))

        scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
        label = pred
        label_num = np.max(label) + 1
        bboxes = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < 800 / (1 * 1):
                continue

            score_i = np.mean(score[label == i])
            if score_i < 0.93:
                continue

            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect) * scale
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))

        bboxes = bboxes.reshape(-1, 4, 2)
        return bboxes
