import torch.nn as nn
import sys
from models import *
from aug import *
from data import *
from utils import *
from torchvision.transforms import transforms
# from tensorboardX import SummaryWriter
import torch.optim as optim
from utils.my_logging import Logging, time
import os
import json
import traceback


class Trainer():
    def __init__(self, ops):
        self.ops = ops
        self.current_path = Path(__file__).parent.absolute().__str__()
        self.best_fitness = 0
        self.epoch = 0
        self.train_loss = 0
        self.eval_loss = 0
        self.train_acc = 0
        self.eval_acc = 0
        
    def initModel(self):
        self.input_channel = self.ops.input_channel
        self.num_classes = self.ops.num_classes
        self.model =  LEDNet50(self.input_channel, self.num_classes) 

    def initTrainFile(self):
        self.train_ratio = self.ops.train_ratio
        if os.path.isfile(self.ops.dataset_path):
            self.data_dir = self.ops.dataset_path
        else:
            self.data_dir = self.ops.data_path+os.sep+self.ops.dataset_name
        assert os.path.exists(self.data_dir), 'The dataset does not exist'

        if not self.ops.label_name:
            labels = os.listdir(self.data_dir)
            labels.sort(key=lambda x:x)
            label2num = dict([(label, idx) for idx, label in enumerate(labels)])
        else:
            label2num = dict([(label, idx) for idx, label in  enumerate(self.ops.label_name)])
        
        data_split = DataSplit(self.data_dir, self.train_ratio, label2num)
        
        self.ops.num_classes = data_split.num_cls()
        
        data_split(self.output_dir)
        # exit()
    
    def initTransfors(self):
        aug_args = self.ops.aug
        self.input_size = self.ops.input_size
        aug_method = []
        if isinstance(aug_args, list):
            for aug in aug_args:
                aug_method.append(method_match(aug))
        if not isinstance(self.input_size, list) and not isinstance(self.input_size, tuple):
            self.input_size = (self.input_size, self.input_size)
        self.train_transforms = transforms.Compose([*aug_method, transforms.Resize(self.input_size), transforms.ToTensor()])
        self.val_transforms = transforms.Compose([transforms.Resize(self.input_size), transforms.ToTensor()])

    def initOutputDir(self):
        self.output_dir = self.ops.output_dir
        self.loging.info(f'Temporary files cache directory:{self.output_dir}')
        os.makedirs(self.output_dir, exist_ok=True)

    def initSummaryWriter(self):
        # self.log_dir = self.ops.log_dir
        # os.makedirs(self.log_dir, exist_ok=True)
        # self.writer = SummaryWriter(self.log_dir)
        # self.loging = Logging(self.log_dir+os.sep+'train_log.log').logger
        self.loging = loging
    
    def initCheckpoint(self):
        if self.ops.save_model not in [None, '']:
            os.makedirs(os.path.abspath(os.path.dirname(self.ops.save_model)), exist_ok=True)
        else:
            self.checkpoint_dir = self.ops.checkpoint_dir
            os.makedirs(self.checkpoint_dir, exist_ok=True)


    def initCriterion(self):
        criterion = self.ops.criterion
        if criterion.lower() == 'crossentropyloss':
            self.criterion = nn.CrossEntropyLoss()

    def initOptimizer(self):
        self.lr = self.ops.lr
        self.adam = self.ops.optim
        if self.adam.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def initTrainLoader(self):
        self.shuffle = self.ops.shuffle
        self.batch_size = self.ops.batch_size
        assert self.batch_size >= 1, 'Batch size must be greater than or equal to 1'
        self.num_workers = self.ops.num_workers
        datagenerator = LedDataGenerator(self.output_dir+os.sep+'train_file.txt', self.train_transforms)
        self.train_dataloader = DataLoader(datagenerator, shuffle=self.shuffle, batch_size=self.batch_size, num_workers=self.num_workers)

    def initValLoader(self):
        batch_size = self.ops.batch_size
        num_workers = self.ops.num_workers
        datagenerator = LedDataGenerator(self.output_dir+os.sep+'eval_file.txt', self.val_transforms)
        self.eval_dataloader = DataLoader(datagenerator, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    def initScheduler(self):
        self.scheduler_name = self.ops.scheduler
        self.step_size = self.ops.step_size
        self.gamma = self.ops.gamma
        if self.scheduler_name.lower() == 'steplr':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

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

    def transformsModel(self):
        device = self.ops.device.lower()
        if device == 'mlu':
            self.model = mlu_quantize.adaptive_quantize(model=self.model, steps_per_epoch=len(self.train_dataloader), bitwidth=16)
            self.model.to(self.deivce)
        elif device == 'gpu':
            self.model = self.model.to(self.deivce)

    def loadPretrainedModel(self):
        if self.ops.pretrained_model not in [None, ""] and os.path.isfile(self.ops.pretrained_model):
            self.model.load_state_dict(torch.load(self.ops.pretrained_model)['model'])

    def transformsOptimizer(self):
        device = self.ops.device.lower()
        if device == 'mlu': 
            self.optimizer = ct.to(self.optimizer, self.deivce)
    
    def saveCheckpoint(self):
        checkpoint = {'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':self.epoch}
        if self.eval_acc >= self.best_fitness:
            self.loging.info(f'Save best weight acc:{self.eval_acc} loss:{self.eval_loss/len(self.eval_dataloader)}')
            self.best_fitness = self.eval_acc
            if self.ops.save_model not in [None, '']:
                weight_save_path = self.ops.save_model
                torch.save(checkpoint, weight_save_path, _use_new_zipfile_serialization=False)
            else:
                weight_save_path = os.path.join(self.checkpoint_dir, "best.pth")
                self.ops.weight = weight_save_path
                torch.save(checkpoint, weight_save_path, _use_new_zipfile_serialization=False)
        # weight_save_path = os.path.join(self.checkpoint_dir, "last.pth")
        # torch.save(checkpoint, weight_save_path, _use_new_zipfile_serialization=False)

    def train(self):
        self.model.train()
        self.train_loss = 0
        predict_res = []
        truth = []
        for iData, iLabel in self.train_dataloader:
           
            self.optimizer.zero_grad()
            outputs = self.model(iData.to(self.deivce))
            loss = self.criterion(outputs, iLabel.to(self.deivce))
            loss.backward()
            self.optimizer.step()
            self.train_loss += loss.data
            predict_res.append(outputs.cpu().detach().numpy())
            truth.append(iLabel.detach().numpy())
        self.train_acc = compute_acc(predict_res, truth)
        self.scheduler.step()
        self.loging.info('Train Epoch:{}[{}/{}({:.0f}%)] train_Loss:{} train_acc:{}'.format(self.epoch, self.epoch, self.epochs, self.epoch/self.epochs*100, self.train_loss / len(self.train_dataloader), self.train_acc))

    def eval(self):
        self.model.eval()
        self.eval_loss = 0
        predict_res = []
        truth = []
        with torch.no_grad():
            for data in self.eval_dataloader:
                iData, iLabel = data
                outputs = self.model(iData.to(self.deivce))
                loss = self.criterion(outputs, iLabel.to(self.deivce))
                self.eval_loss += loss.data
                predict_res.append(outputs.cpu().detach().numpy())
                truth.append(iLabel.detach().numpy())
            self.eval_acc = compute_acc(predict_res, truth)
            self.loging.info('Eval eval_Loss:{} eval_acc:{}'.format(self.eval_loss / len(self.eval_dataloader), self.eval_acc))
            

    def startTransplant(self):
        self.loging.info("Start transplant")
        cmd_params = f'python {self.current_path}/utils/transplant_plugin.py '
        for key,value in self.ops.__dict__.items():
            cmd_params += f'--{key} {value} '
        self.loging.info(f'{cmd_params}')
        os.system(cmd_params)

    def process(self):
        device = self.ops.device.lower()
        if device == 'mlu':
            ct.set_cnml_enabled(False)
        self.epochs = self.ops.epochs
        assert self.epochs >= 1, 'Epochs must be greater than or equal to 1'
        for self.epoch in range(1, self.epochs+1):
            self.train()
            self.eval()
            self.saveCheckpoint()
        # self.startTransplant()

    def worker(self):
        self.initSummaryWriter()
        self.initOutputDir()
        self.initCheckpoint()
        self.initTransfors()
        self.initTrainFile()
        self.initTrainLoader()
        self.initValLoader()
        self.initDevice()
        self.initModel()
        self.transformsModel()
        self.loadPretrainedModel()
        self.initCriterion()
        self.initOptimizer()
        self.transformsOptimizer()
        self.initScheduler()
        self.process()

def main(ops):
    ops.data_path = os.path.dirname(ops.dataset_path)
    ops.dataset_name = os.path.basename(ops.dataset_path)
    if ops.device.lower() == 'mlu':
        import torch_mlu.core.mlu_quantize as mlu_quantize
        import torch_mlu.core.mlu_model as ct
    trainer = Trainer(ops)
    trainer.worker()
    
if __name__ == "__main__":
    ops = parse_args()
    os.makedirs(ops.log_dir, exist_ok=True)
    loging = Logging(ops.log_dir+os.sep+'log.txt').logger
    try:
        main(ops)
    except Exception as e:
        loging.info(traceback.format_exc())