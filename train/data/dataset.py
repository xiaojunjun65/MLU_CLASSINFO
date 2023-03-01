import imghdr
from posixpath import join
from torch.utils.data import Dataset, DataLoader, dataloader
from torchvision import transforms
import random
import os
from PIL import Image
import torch
import json

class DataSplit():
    def __init__(self, data_root, split_ratio, label2num):
        self.label2num = label2num
        self.data_root = data_root
        self.split_ratio = split_ratio
        self.data_file_name = {}
        self.train_data = {}
        self.eval_data = {}
        if os.path.isfile(self.data_root):
            self.data_cls = list(self.label2num.keys())
        else:
            self.data_cls = os.listdir(self.data_root)
            for cls in self.data_cls:
                assert os.path.exists(os.path.join(self.data_root, cls)), 'The label folder does not exist'
                
    def __call__(self, path):
        self.load_data()
        self.split_data()
        with open(os.path.join(path, "train_file.txt"), 'w') as f:
            for key, files in self.train_data.items():
                for file in files:
                    f.write(file+","+str(self.label2num[key])+"\n")

        with open(os.path.join(path, "eval_file.txt"), 'w') as f:
            for key, files in self.eval_data.items():
                for file in files:
                    f.write(file+","+str(self.label2num[key])+"\n")
        return self.train_data, self.eval_data


    def load_data(self):
        if os.path.isfile(self.data_root):
            dir_name = os.path.dirname(self.data_root)
            for cls in self.data_cls:
                self.data_file_name[cls] = []
            with open(self.data_root) as fp:
                label_info = fp.readlines()
            for item in label_info:
                if len(item.strip().split(',')) == 2:
                    img_file, cls = item.strip().split(',')
                    self.data_file_name[cls].append(os.path.join(dir_name, img_file))
        else:
            for cls in self.data_cls:
                if not cls in list(self.label2num.keys()):
                    continue
                cls_path = os.path.join(self.data_root, cls)
                image_path = []
                for name in os.listdir(cls_path):
                    image_path.append(os.path.join(cls_path, name))
                self.data_file_name[cls] = image_path
            
    def split_data(self):
        for key, vals in self.data_file_name.items():
            traind = set(random.sample(vals, int(self.split_ratio * len(vals))))
            testd = list(set(vals) - traind)
            self.train_data[key] = list(traind) if len(traind) > 0 else testd
            self.eval_data[key] = testd

    def num_cls(self):
        return len(self.label2num.keys())


class DataLoad():
    def __init__(self,train_data_path,eval_data_path, label2num):
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.label2num = label2num
        self.train_dataset = {}
        self.eval_dataset = {}

    def load_data(self):
        class_dir = os.listdir(self.train_data_path)
        for i in class_dir:
            class_path = os.path.join(os.path.abspath(self.train_data_path), i)
            train_data_file = os.listdir(class_path)
            image_path = []
            for file_name in train_data_file:
                image_path.append(os.path.join(class_path,file_name))
            self.train_dataset[i] = image_path
        class_dir = os.listdir(self.eval_data_path)
        for i in class_dir:
            class_path = os.path.join(os.path.abspath(self.eval_data_path), i)
            eval_data_file = os.listdir(class_path)
            image_path = []
            for file_name in eval_data_file:
                image_path.append(os.path.join(class_path,file_name))
            self.eval_dataset[i] = image_path

    def write_txt_file(self,path):
        self.load_data()
        with open(os.path.join(path, "train_file_0630.txt"), 'w') as f:
            for key, files in self.train_dataset.items():
                for file in files:
                    f.write(file+","+str(self.label2num[key])+"\n")

        with open(os.path.join(path, "eval_file_0709.txt"), 'w') as f:
            for key, files in self.eval_dataset.items():
                for file in files:
                    f.write(file+","+str(self.label2num[key])+"\n")

class LedDataGenerator(Dataset):
    def __init__(self, data_file, transform):
        self.data_file = data_file
        self.transform = transform
        self.dataset = []
        self.labals = []
        self.access_idx = []
        self.data_init()
    
    def __getitem__(self,index):
        image = Image.open(self.dataset[index])
        # print(self.dataset[index])
        # print(image.mode)
        label = self.labals[index]
        tf_image = self.transform(image).type(torch.FloatTensor)
        return tf_image, label          

    def __len__(self):
        return len(self.dataset)

    def data_init(self):
        assert os.path.exists(self.data_file)
        okNum = 0
        qibaoNum = 0
        with open(self.data_file, 'r') as f:
            data_file = f.readlines()
        for file in data_file:
            label = file.strip().split(",")[-1]
            val = file.strip()[:-2]
            if label == '1':
                okNum += 1
            else:
                qibaoNum += 1
            self.labals.append(int(label))
            self.dataset.append(val)

if __name__ == "__main__":
    mydataload = DataLoad('../dataset/images0630/images0630','../dataset/测试图片20210709/测试图片20210709')
    mydataload.write_txt_file('../dataset')
