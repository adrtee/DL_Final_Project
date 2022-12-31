import re
import cv2
import os,argparse
import pytesseract
from PIL import Image
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt

class number_detection():
    def __init__(self, img = None):
        self.image = img
        self.angle = 0
    def get_input(self):
        self.image = cv2.imread("./0004.png")
    def preprocessing(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # plt.imshow(self.image)
        # plt.show()
        self.image = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # plt.imshow(self.image)
        # plt.show()
        
        # print(self.image.dtype)
        # self.image = exposure.equalize_hist(self.image)
        # self.image = np.clip(self.image, 0, 255).astype(np.uint8)
        # print(self.image.dtype)
    def get_text(self):
        custom_config = '-l eng --oem 3 --psm 6 '
        self.image = Image.fromarray(self.image)
        data = pytesseract.image_to_string(self.image, config=custom_config)
        
        while len(re.findall(r'\d+', data)) == 0 and self.angle<=360:
            self.angle += 15
            self.image = self.image.rotate(self.angle)
            data = pytesseract.image_to_string(self.image, config=custom_config)
            # print(data)

        if len(re.findall(r'\d+', data)) > 0:
            data = re.findall(r'\d+', data)
            max_num = max(data, key=int)
            min_num = min(data, key=int)
        return max_num, min_num 

if __name__=='__main__':
    nd = number_detection()
    nd.get_input()
    # nd.preprocessing()
    max_num, min_num = nd.get_text()
    # text = re.findall(r'\d+', text)
    print(min_num)
    print(max_num)
    # print(max(text, key=int))
