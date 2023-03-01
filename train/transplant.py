import argparse
import shutil
import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct
from models import *
from utils import *
from data import *


def str2bool(v):
        return v.lower() in ("true", "t", "1")

class Transplanter:
    def __init__(self, ops):
        self.ops = ops

    def iniTransfors(self):
        self.input_size = self.ops.input_size
        if not isinstance(self.input_size, list) and not isinstance(self.input_size, tuple):
            self.input_size = (self.input_size, self.input_size)
        self.transforms = transforms.Compose([transforms.Resize(self.input_size), transforms.ToTensor()])

    def initNumClasses(self):
        self.output_dir = self.ops.output_dir
        train_file = self.output_dir+os.sep+'train_file.txt'
        with open(train_file) as fp:
            data = fp.readline()
        image_file = data.split(',')[0]
        dataset_path = os.path.abspath(os.path.dirname(image_file) + os.path.sep + "..")
        self.ops.num_classes = len(os.listdir(dataset_path))

    def initModel(self):
        self.input_channel = self.ops.input_channel
        self.num_classes = self.ops.num_classes
        self.orig_model =  LEDNet50(self.input_channel, self.num_classes).eval().float()
        
        device = self.ops.device.lower()
        if device == 'mlu':
            self.model = mlu_quantize.adaptive_quantize(self.orig_model, steps_per_epoch=1, bitwidth=16, inplace=True)
        else:
            self.model = self.orig_model
        if self.ops.model in [None, ""] or not os.path.isfile(self.ops.model):
            self.ops.model = self.ops.output_dir + os.sep + "best.pth"
        if os.path.isfile(self.ops.model):
            self.model.load_state_dict(torch.load(self.ops.model, map_location=torch.device('cpu'))['model'])
        if device == 'mlu':
            self.model = mlu_quantize.dequantize(self.model)
        self.weight_name = self.ops.model.split('/')[-1].split('.')[0]

    def readImage(self):
        with open(self.ops.output_dir+os.sep+'train_file.txt') as f:
            data_file = f.readlines()

        self.image_list = [i.strip().split(',')[0] for i in data_file]

    def generateQuantizeModel(self):
        assert self.ops.quantized_dir
        os.makedirs(self.ops.quantized_dir, exist_ok=True)

        if self.ops.quantized_mode == 0:
            quantized_mode = 'int8'
        else:
            quantized_mode = 'int16'

        qconfig = {'iteration': self.ops.iteration, 'use_avg':self.ops.use_avg, 'data_scale':self.ops.data_scale, 'mean': self.ops.mean, 'std': self.ops.std, 'per_channel': self.ops.per_channel, 'firstconv': self.ops.firstconv}
        self.quantize_model = mlu_quantize.quantize_dynamic_mlu(self.model, qconfig_spec=qconfig, dtype=quantized_mode, gen_quant=True)
        for image_file in self.image_list[:self.ops.iteration]:
            img = Image.open(image_file)
            idata = self.transforms(img).unsqueeze(0)
            self.quantize_model(idata)
        self.quantize_weight_path = self.ops.quantized_dir+os.sep+f'{self.weight_name}-{quantized_mode}.pth'
        torch.save(self.quantize_model.state_dict(), self.quantize_weight_path)

    def loadQuantizeModel(self):
        self.quantize_model = mlu_quantize.quantize_dynamic_mlu(self.orig_model)
        self.quantize_model.load_state_dict(torch.load(self.quantize_weight_path))

    def generateOfflineModel(self):
        assert self.ops.offline_dir
        if self.ops.offline_model not in [None, '']:
            os.makedirs(os.path.abspath(os.path.dirname(self.ops.offline_model)), exist_ok=True)
        else:
            os.makedirs(self.ops.offline_dir, exist_ok=True)

        if self.ops.core_number < 1:
            self.ops.core_number = 1
        if self.ops.core_version == 'MLU220' and self.ops.core_number > 4:
            self.ops.core_number = 4
        if self.ops.core_version == 'MLU270' and self.ops.core_number > 16:
            self.ops.core_number = 16

        ct.set_cnml_enabled(True)
        ct.set_device(-1)
        ct.set_core_version(self.ops.core_version)
        ct.set_core_number(self.ops.core_number)

        if self.ops.offline_model not in [None, '']:
            ct.save_as_cambricon(self.ops.offline_model.split('.cambricon')[0])
        else:
            ct.save_as_cambricon(f'{self.ops.offline_dir}/{self.weight_name}-{self.ops.core_version.lower()}')
        
        idata = torch.randn(self.ops.batch_size, self.ops.input_channel, self.ops.input_size, self.ops.input_size).float()
        self.fuse_model = torch.jit.trace(self.quantize_model.to(ct.mlu_device()), idata.to(ct.mlu_device()), check_trace=False)
        with torch.no_grad():
            self.fuse_model(idata.to(ct.mlu_device()))
        ct.save_as_cambricon('')
    
    def worker(self):
        self.iniTransfors()
        if self.ops.num_classes in [None, '']:
            self.initNumClasses()
        self.initModel()
        self.readImage()
        self.generateQuantizeModel()
        self.loadQuantizeModel()
        self.generateOfflineModel()
        shutil.rmtree(self.ops.quantized_dir)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/workspace/volume/guojun/Train/Classification/outputs', help='文件输出路径')
    parser.add_argument('--quantized_dir', type=str, default='/workspace/quantized', help='量化模型保存地址')
    parser.add_argument('--offline_dir', type=str, default='./offline', help='离线模型保存地址')
    parser.add_argument("--input_channel", type=int, default=3, help="模型输入图片的通道数")
    parser.add_argument("--input_size", type=int, default=100, help="输入网络模型的尺寸")
    parser.add_argument("--num_classes", type=int, required=True, help="模型的类别数")
    parser.add_argument("--model", type=str, required=True, help="模型文件")
    parser.add_argument("--device", type=str, default="mlu", help="模型文件在什么设备训练")

    #quantize config
    parser.add_argument('--iteration', type=int, default=1, help='使用多少张图片量化')
    parser.add_argument("--use_avg", type=str2bool, default='false', help ="是否使用均值")
    parser.add_argument('--data_scale', type=float, default=1.0, help='图片缩放')
    parser.add_argument('--mean', nargs='+', default=[0,0,0], help='firstconv使用')
    parser.add_argument('--std', nargs='+', default=[1,1,1], help='firstconv使用')
    parser.add_argument('--per_channel', type=str2bool, default='false', help='通道量化')
    parser.add_argument('--firstconv', type=str2bool, default='false', help='使用firstconv')
    parser.add_argument("--quantized_mode", type=int, default=1, choices=[0, 1], help ="0-int8 1-int16")

    #offline
    parser.add_argument('--batch_size', type=int, default=1, help='生成多少batchsize的离线模型')
    parser.add_argument('--core_version', type=str, default='MLU220', help='生成什么MLU环境的离线模型')
    parser.add_argument('--core_number', type=int, default=4, help='生成多少核的离线模型')
    parser.add_argument('--offline_model', type=str, default=None, help='离线模型保存路径')

    ops = parser.parse_args()

    transplanter = Transplanter(ops)
    transplanter.worker()