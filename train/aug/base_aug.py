import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance

class AugBase(object):
    def __init__(self):
        self.input_len = 1
    
    def __call__(self,img):
        img = self.auditInput(img)

        if isinstance(img, list):

            if self.input_len > 1:
                return self.forward(img)

            if self.input_len == 1:
                img[0] = self.forward(img[0])
                return img

        return self.forward(img)
    
    def forward(self, img):
        pass

    def auditImg(self, img):
        pass

    def auditInput(self, img):
        if isinstance(img, tuple):
            img = list(img)

        if isinstance(img, list):
            img[0] = self.auditImg(img[0])
            return img

        if self.input_len == 1:
            return self.auditImg(img)

    @staticmethod
    def pillow2cv(pillow_img, is_rgb2bgr=True):
        cv_image = np.array(pillow_img)
        if is_rgb2bgr:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        return cv_image

    @staticmethod
    def cv2pillow(cv_img, is_bgr2rgb=True):
        if is_bgr2rgb:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_img)

    @staticmethod
    def isPil(img):
        return isinstance(img, Image.Image)

    @staticmethod
    def isNumpy(img):
        return isinstance(img, np.ndarray)

class PilAugBase(AugBase):
    def auditImg(self, img):
        if self.isNumpy(img):
            img = self.cv2pillow(img)

        if self.isPil(img):
            return img

# 随机颜色扰动
class ColorJitterAug(PilAugBase):
    def forward(self, img):
        return ImageEnhance.Color(img).enhance(np.random.uniform(0.8, 1.3))

class BrightnessJitterAug(PilAugBase):
    def forward(self, img):
        return ImageEnhance.Brightness(img).enhance(np.random.uniform(0.6, 1.5))

class ContrastJitterAug(PilAugBase):
    def forward(self, img):
        return ImageEnhance.Contrast(img).enhance(np.random.uniform(0.5, 1.8))

class RandomColorJitterAug(PilAugBase):
    def forward(self, img):
        if random.randint(0, 1):
            img = ImageEnhance.Color(img).enhance(np.random.uniform(0.8, 1.3))
        if random.randint(0, 1):
            img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.6, 1.5))
        if random.randint(0, 1):
            img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.5, 1.8))
        return img


def method_match(aug_name):
    if aug_name.lower() == 'randomcolorcitter':
        return RandomColorJitterAug()