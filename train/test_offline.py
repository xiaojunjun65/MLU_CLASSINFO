import argparse
import shutil
import os
import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct
from models import *
from utils import *
from data import *

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

def str2bool(v):
        return v.lower() in ("true", "t", "1")

class OfflineTester:
    def __init__(self, ops):
        self.ops = ops
        os.makedirs(self.ops.output_path, exist_ok=True)
        
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
            self.ops.model = self.output_dir + os.sep + "best.pth"
        if os.path.isfile(self.ops.model):
            self.model.load_state_dict(torch.load(self.ops.model, map_location=torch.device('cpu'))['model'])
        if device == 'mlu':
            self.model = mlu_quantize.dequantize(self.model)
        self.weight_name = self.ops.model.split('/')[-1].split('.')[0]

    def readImage(self):
        # if not self.ops.label_name:
        #     labels = os.listdir(self.ops.dataset_path)
        #     labels.sort(key=lambda x:x)
        #     self.label2num = dict([(label, idx) for idx, label in enumerate(labels)])
        # else:
        #     self.label2num = dict([(label, idx) for idx, label in  enumerate(self.ops.label_name)])
        self.num2label = dict([(idx, label) for idx, label in  enumerate(self.ops.label_name)])

        self.image_list = [os.path.join(self.ops.dataset_path, file) for file in os.listdir(self.ops.dataset_path) if file.split('.')[-1].lower() in IMG_FORMATS]


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

    def testOfflineModel(self):
        if self.ops.core_number < 1:
            self.ops.core_number = 1
        if self.ops.core_version == 'MLU220' and self.ops.core_number > 4:
            self.ops.core_number = 4
        if self.ops.core_version == 'MLU270' and self.ops.core_number > 16:
            self.ops.core_number = 16

        ct.set_cnml_enabled(True)
        ct.set_core_version(self.ops.core_version)
        ct.set_core_number(self.ops.core_number)
        
        idata = torch.randn(self.ops.batch_size, self.ops.input_channel, self.ops.input_size, self.ops.input_size).float()
        self.fuse_model = torch.jit.trace(self.quantize_model.to(ct.mlu_device()), idata.to(ct.mlu_device()), check_trace=False)

        fp = open(os.path.join(self.ops.output_path, 'result.txt'), 'w')
        # true_num = 0
        for image_file in self.image_list:
            image = Image.open(image_file)
            image = self.transforms(image).unsqueeze(0)

            with torch.no_grad():
                outputs = self.fuse_model(image.to(ct.mlu_device()))
                predict_res = outputs.cpu().detach().numpy()
                res = np.argmax(predict_res, axis=1)[0]
                fp.write('{},{}\n'.format(image_file, self.num2label[res]))
        fp.close()


    def worker(self):
        self.iniTransfors()
        if self.ops.num_classes in [None, '']:
            # self.initNumClasses()
            self.ops.num_classes = len(self.ops.label_name)
        self.initModel()
        self.readImage()
        self.generateQuantizeModel()
        self.loadQuantizeModel()
        self.testOfflineModel()
        shutil.rmtree(self.ops.quantized_dir)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./runs', help='文件输出路径')
    parser.add_argument('--quantized_dir', type=str, default='/workspace/quantized', help='量化模型保存地址')  # ./quantized
    parser.add_argument("--input_channel", type=int, default=3, help="模型输入图片的通道数")
    parser.add_argument("--input_size", type=int, default=100, help="输入网络模型的尺寸")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集地址")
    parser.add_argument("--label_name", nargs="+", required=True, help="类别名称到数字映射")
    parser.add_argument("--num_classes", type=int, help="模型的类别数")
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
    
    parser.add_argument("--output_path", type=str, required=True, help="测试结果保存地址")
    ops = parser.parse_args()

    offline_tester = OfflineTester(ops)
    offline_tester.worker()