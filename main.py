from DocumentDetector.psenet.detector import Detector
from DocumentRecognizer.crnn.recognizer import Recognizer
import cv2
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

doc_detector = Detector(device)
doc_detector.load_weights('model path')

doc_recognizer = Recognizer(device)
doc_recognizer.load_weights('model path')

img_path = 'img path'
src_img = cv2.imread(img_path)
bboxes = doc_detector.predict(src_img)
cv2.imshow("src", src_img)

for box in bboxes:
    points = box.reshape(4, 2)
    img = src_img[points[1, 1]:points[0, 1], points[1, 0]:points[2, 0]]
    res = doc_recognizer.predict(img)
    print(res)
    cv2.imshow("part_img", img)
    cv2.waitKey(0)
