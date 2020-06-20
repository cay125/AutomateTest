from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch


class ChineseTextLoader(Dataset):
    def __init__(self, converter, img_root, label_path, alphabet, resize, transforms=None):
        super(ChineseTextLoader, self).__init__()
        self.converter = converter
        self.img_root = img_root
        self.labels = self.get_labels(label_path)
        self.alphabet = alphabet
        self.transforms = transforms
        self.width, self.height = resize

    # print(list(self.labels[1].values())[0])
    def get_labels(self, label_path):
        with open(label_path, 'r', encoding='utf-8') as file:
            labels = []
            for c in file.readlines():
                text = c.split(' ')[-1][:-1]
                flag = True
                for item in text:
                    if item not in self.converter.dict.keys():
                        flag = False
                        break
                if flag:
                    labels.append({c.split(' ')[0]: c.split(' ')[-1][:-1]})
        return labels

    def __len__(self):
        return len(self.labels)

    def preprocessing(self, image):
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image.sub_(0.5).div_(0.5)

        return image

    def __getitem__(self, index):
        image_name = list(self.labels[index].keys())[0]
        # label = list(self.labels[index].values())[0]
        image = cv2.imread(self.img_root + '/' + image_name)
        # print(self.img_root+'/'+image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        # Data augmentation
        # width > len ==> resize width to len
        # width < len ==> padding width to len
        # if self.isBaidu:
        # 	# image = self.compensation(image)
        # 	image = cv2.resize(image, (0,0), fx=160/w, fy=32/h, interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (0, 0), fx=self.width / w, fy=self.height / h, interpolation=cv2.INTER_CUBIC)
        image = (np.reshape(image, (32, self.width, 1))).transpose(2, 0, 1)
        image = self.preprocessing(image)

        return image, index


if __name__ == '__main__':
    pass
