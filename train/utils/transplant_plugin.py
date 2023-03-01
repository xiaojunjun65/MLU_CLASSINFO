import sys
import os 
from pathlib import Path
sys.path.append(Path(__file__).parent.absolute().__str__()) 
sys.path.append(Path(__file__).parent.parent.absolute().__str__())
from models import *
from utils import *
from data import *
import glob
try:
    import torch_mlu.core.mlu_quantize as mlu_quantize
    import torch_mlu.core.mlu_model as ct
except:
    pass
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

class AutoTransplant():
    def __init__(self, ops):
        self.ops = ops

    def initModel(self):
        self.input_channel = self.ops.input_channel
        self.num_classes = self.ops.num_classes
        self.orig_model =  LEDNet50(self.input_channel, self.num_classes).eval().float()
        self.model = mlu_quantize.adaptive_quantize(self.orig_model, steps_per_epoch=1, bitwidth=16, inplace=True)
        assert self.ops.weight
        self.weight_name = '.'.join(self.ops.weight.split('/')[-1].split('.')[:-1])
        self.model.load_state_dict(torch.load(self.ops.weight)['model'])
        self.model = mlu_quantize.dequantize(self.model)

    def loadQuantizeModel(self):
        self.quantize_model = mlu_quantize.quantize_dynamic_mlu(self.orig_model)
        self.quantize_model.load_state_dict(torch.load(self.quantize_weight_path))

    def iniTransfors(self):
        self.input_size = self.ops.input_size
        if not isinstance(self.input_size, list) and not isinstance(self.input_size, tuple):
            self.input_size = (self.input_size, self.input_size)
        self.transforms = transforms.Compose([transforms.Resize(self.input_size), transforms.ToTensor()])

    def readImage(self):
        with open(self.ops.output_dir+os.sep+'train_file.txt') as f:
            data_file = f.readlines()

        image_list = [i.strip().split(',')[0] for i in data_file]
        self.images = image_list[:self.ops.iteration]

    def generateQuantizeModel(self):
        assert self.ops.quantized_dir
        os.makedirs(self.ops.quantized_dir, exist_ok=True)

        if self.ops.quantized_mode:
            quantized_mode = 'int16'
        else:
            quantized_mode = 'int8'

        qconfig = {'iteration': self.ops.iteration, 'use_avg':self.ops.use_avg, 'data_scale':self.ops.data_scale, 'mean': self.ops.mean, 'std': self.ops.std, 'per_channel': self.ops.per_channel, 'firstconv': self.ops.firstconv}
        self.quantize_model = mlu_quantize.quantize_dynamic_mlu(self.model, qconfig_spec=qconfig, dtype=quantized_mode, gen_quant=True)
        for image_file in self.images:
            img = Image.open(image_file)
            idata = self.transforms(img).unsqueeze(0)
            self.quantize_model(idata)
        self.quantize_weight_path = self.ops.quantized_dir+os.sep+f'{self.weight_name}-{quantized_mode}.pth'
        torch.save(self.quantize_model.state_dict(), self.quantize_weight_path)

    def generate270OfflineModel(self):
        if self.ops.core_number_270 < 0:return
        ct.set_cnml_enabled(True)
        assert self.ops.offline_dir
        os.makedirs(self.ops.offline_dir, exist_ok=True)
        ct.set_device(-1)
        ct.set_core_number(self.ops.core_number_270)
        ct.set_core_version("MLU270")
        ct.save_as_cambricon(f'{self.ops.offline_dir}/{self.weight_name}-mlu270')
        img = Image.open(self.images[0])
        idata = self.transforms(img).type(torch.FloatTensor).unsqueeze(0).to(ct.mlu_device())
        self.fuse_model = torch.jit.trace(self.quantize_model.to(ct.mlu_device()), idata.to(ct.mlu_device()), check_trace=False)
        with torch.no_grad():
            self.fuse_model(idata.to(ct.mlu_device()))
        ct.save_as_cambricon('')

    def generate220OfflineModel(self):
        if self.ops.core_number_220 < 0:return
        ct.set_cnml_enabled(True)
        assert self.ops.offline_dir
        os.makedirs(self.ops.offline_dir, exist_ok=True)
        ct.set_device(-1)
        ct.set_core_number(self.ops.core_number_220)
        ct.set_core_version("MLU220")
        ct.save_as_cambricon(f'{self.ops.offline_dir}/{self.weight_name}-mlu220')
        img = Image.open(self.images[0])
        idata = self.transforms(img).type(torch.FloatTensor).unsqueeze(0).to(ct.mlu_device())
        self.fuse_model = torch.jit.trace(self.quantize_model.to(ct.mlu_device()), idata.to(ct.mlu_device()), check_trace=False)
        with torch.no_grad():
            self.fuse_model(idata.to(ct.mlu_device()))
        ct.save_as_cambricon('')

    def worker(self):
        self.readImage()
        self.initModel()
        self.iniTransfors()
        self.generateQuantizeModel()
        self.loadQuantizeModel()
        self.generate270OfflineModel()
        self.generate220OfflineModel()

def main(ops):
    atp = AutoTransplant(ops)
    atp.worker()

def create_quantize_model(ops):
    atp = AutoTransplant(ops)
    atp.readImage()
    atp.initModel()
    atp.iniTransfors()
    atp.generateQuantizeModel()

def create_offline_model(ops):
    atp = AutoTransplant(ops)
    atp.readImage()
    atp.initModel()
    atp.iniTransfors()
    if atp.ops.quantized_mode:
        quantized_mode = 'int16'
    else:
        quantized_mode = 'int8'
    atp.quantize_weight_path = atp.ops.quantized_dir+os.sep+f'{atp.weight_name}-{quantized_mode}.pth'
    atp.loadQuantizeModel()
    atp.generate270OfflineModel()
    atp.generate220OfflineModel()

if __name__ == "__main__":
    main(parse_args())