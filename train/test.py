import argparse
import os
from torchvision.transforms import transforms
from models import *
from aug import *
from data import *
from utils import *
from utils.my_logging import Logging


def str2bool(v):
    return v.lower() in ("true", "t", "1")

class Tester:
    def __init__(self, ops):
        self.ops = ops
        self.loging = Logging(self.ops.log_dir+os.sep+'test_log.log').logger

    def initOutputDir(self):
        self.output_dir = self.ops.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def initTransfors(self):
        self.input_size = self.ops.input_size
        if not isinstance(self.input_size, list) and not isinstance(self.input_size, tuple):
            self.input_size = (self.input_size, self.input_size)
        self.transforms = transforms.Compose([transforms.Resize(self.input_size), transforms.ToTensor()])

    def initTestLoader(self):
        batch_size = self.ops.batch_size
        num_workers = self.ops.num_workers
        datagenerator = LedDataGenerator(self.output_dir+os.sep+'eval_file.txt', self.transforms)
        self.test_dataloader = DataLoader(datagenerator, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        if self.ops.num_classes in [None, '']:
            self.initNumClasses()

    def initNumClasses(self):
        eval_file = self.output_dir+os.sep+'eval_file.txt'
        with open(eval_file) as fp:
            data = fp.readline()
        image_file = data.split(',')[0]
        dataset_path = os.path.abspath(os.path.dirname(image_file) + os.path.sep + "..")
        self.ops.num_classes = len(os.listdir(dataset_path))

    def initDevice(self):
        device = self.ops.device.lower()
        if device == 'gpu':
            if torch.cuda.is_available():
                self.deivce = torch.device('cuda')
        elif device == 'mlu':
            if torch.is_mlu_available():
                ct.set_cnml_enabled(False)
                self.deivce = torch.device('mlu')
        else:
            self.deivce = torch.device('cpu')

    def initModel(self):
        self.input_channel = self.ops.input_channel
        self.num_classes = self.ops.num_classes
        self.model =  LEDNet50(self.input_channel, self.num_classes)

    def transformsModel(self):
        device = self.ops.device.lower()
        if device == 'mlu':
            self.model = mlu_quantize.adaptive_quantize(model=self.model, steps_per_epoch=len(self.test_dataloader), bitwidth=16)
            self.model.to(self.deivce)
        elif device == 'gpu':
            self.model = self.model.to(self.deivce)

    def loadPretrainedModel(self):
        if self.ops.model in [None, ""] or not os.path.isfile(self.ops.model):
            self.ops.model = self.output_dir + os.sep + "best.pth"
        if os.path.isfile(self.ops.model):
            self.model.load_state_dict(torch.load(self.ops.model)['model'])

    def test(self):
        self.model.eval()
        predict_res = []
        truth = []
        with torch.no_grad():
            for data in self.test_dataloader:
                iData, iLabel = data
                print(iData)
                outputs = self.model(iData.to(self.deivce))
                print(outputs.cpu().detach().numpy())
                predict_res.append(outputs.cpu().detach().numpy())
                truth.append(iLabel.detach().numpy())
            test_acc = compute_acc(predict_res, truth)
            self.loging.info('Test test_acc:{}'.format(test_acc))
            output_dir = os.path.dirname(self.ops.model)
            with open(os.path.join(output_dir, 'result.json'), 'w') as fp:
                fp.write('Test test_acc:{}'.format(test_acc))
        
    def worker(self):
        self.initOutputDir()
        self.initTransfors()
        self.initTestLoader()
        self.initDevice()
        self.initModel()
        self.transformsModel()
        self.loadPretrainedModel()
        self.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default='/workspace/volume/guojun/Train/Classification/outputs', help="log输出路径")  # ./runs
    parser.add_argument("--output_dir", type=str, default='/workspace/volume/guojun/Train/Classification/outputs', help="文件输出路径")  # ./runs
    parser.add_argument("--device", type=str, default='mlu', help="需要使用什么硬件测试网络")
    parser.add_argument("--input_channel", type=int, default=3, help="模型输入图片的通道数")
    parser.add_argument("--input_size", type=int, default=100, help="输入网络模型的尺寸")
    parser.add_argument("--num_classes", type=int, required=True, help="模型的类别数")
    parser.add_argument("--model", type=str, help="模型文件")
    # parser.add_argument("--dataset_path", type=str, required=True, help="数据集地址")
    parser.add_argument("--batch_size", type=int, default=12, help="批次数")
    parser.add_argument("--num_workers", type=int, default=16, help="多少个进程处理数据")
    ops = parser.parse_args()

    if ops.device.lower() == 'mlu':
        import torch_mlu.core.mlu_quantize as mlu_quantize
        import torch_mlu.core.mlu_model as ct
    
    os.makedirs(ops.log_dir, exist_ok=True)

    tester = Tester(ops)
    tester.worker()
    