from DocumentRecognizer.crnn.crnnNets import *
from Common.utils import alphabetChinese
from PIL import Image
import cv2


class Recognizer:
    def __init__(self, device=torch.device('cpu')):
        alphabet = alphabetChinese
        nclass = len(alphabet) + 1
        self.model = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=True, GPU=False, alphabet=alphabet)
        self.device = device

    def load_weights(self, model_path: str):
        self.model.load_weights(model_path)
        self.model = self.model.to(self.device)

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        res = self.model.predict(img, self.device)
        return res
