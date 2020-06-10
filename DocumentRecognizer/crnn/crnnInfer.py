import cv2
import torch
from PIL import Image
from crnnNets import *
from Common.utils import alphabetChinese

alphabet = alphabetChinese
chi_dict = {}
for i, char in enumerate(alphabet):
    if char in chi_dict.keys():
        chi_dict[char + char] = i + 1
    else:
        chi_dict[char] = i + 1
# torch.save(chi_dict, 'new_chinese_dict')
nclass = len(alphabet) + 1
ocr = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=True, GPU=False, alphabet=alphabet)
state_dict = torch.load('c:/users/xiangpu/downloads/ocr-lstm.pth', map_location='cpu')
ocr.load_weights('c:/users/xiangpu/downloads/ocr-lstm.pth')
img_path = 'C:/Users/xiangpu/Documents/Tencent Files/793987544\FileRecv/MobileFile/screens/results/'
for idx in range(42):
    img = cv2.imread(img_path + 'part_' + str(
        idx) + '_of_42_pseNet_2020_06_03_14_49_Screenshot_2020-05-29-12-49-00-274_com.miui.playe.png', 0)
    cv2.imshow("src", img)
    img = Image.fromarray(img)
    print(img.size)
    res = ocr.predict(img)
    print(res)
    cv2.waitKey(0)
