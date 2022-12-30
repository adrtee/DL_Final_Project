import re
import cv2
import os,argparse
import pytesseract
from PIL import Image

class number_detection():
    def __init__(self, img = None):
        self.image = img
        self.angle = 0
    def get_input(self):
        self.image = cv2.imread("./aaa.png")
    def preprocessing(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    def get_text(self):
        custom_config = '-l eng --oem 3 --psm 6 '
        self.image = Image.fromarray(self.image)
        data = pytesseract.image_to_string(self.image, config=custom_config)
        
        while len(re.findall(r'\d+', data)) == 0 and self.angle<=360:
            self.angle += 15
            self.image = self.image.rotate(self.angle)
            data = pytesseract.image_to_string(self.image, config=custom_config)

        if len(re.findall(r'\d+', data)) > 0:
            data = re.findall(r'\d+', data)
            data = max(data, key=int)
        return data

nd = number_detection()
nd.get_input()
nd.preprocessing()
text = nd.get_text()
# text = re.findall(r'\d+', text)
print(text)
# print(max(text, key=int))
