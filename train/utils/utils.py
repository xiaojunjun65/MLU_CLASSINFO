import argparse
import os
from pathlib import Path
def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")
    parser = argparse.ArgumentParser()
    # current_path = Path(__file__).parent.parent.absolute().__str__()
    #output path
    parser.add_argument("--output_dir", type=str, default='/workspace/volume/guojun/Train/Classification/outputs', help="文件输出路径")  # ./runs
    parser.add_argument("--log_dir", type=str, default='/workspace/volume/guojun/Train/Classification/outputs', help="log输出路径")  # ./runs
    parser.add_argument("--checkpoint_dir", type=str, default='./runs', help="模型保存地址")
    #device
    parser.add_argument("--device", type=str, default='mlu', help="需要使用什么硬件训练网络")

    #net params
    # parser.add_argument("--network_name", type=str, required=True, help="要训练的网络名称")
    parser.add_argument("--input_channel", type=int, default=3, help="训练图片的通道数")
    parser.add_argument("--num_classes", type=int, help="要训练的类别数")
    
    #data augment params
    parser.add_argument("--aug", nargs="+", default=['randomcolorcitter'], help="需要增强的方法")
    parser.add_argument("--input_size", type=int, default=100, help="输入网络模型的尺寸")
    
    #Criterion
    parser.add_argument("--criterion", type=str, default='crossentropyloss', help="损失函数")

    #Optimizer
    parser.add_argument("--optim", type=str, default='adam', help="优化器")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")

    #Scheduler
    parser.add_argument("--scheduler", type=str, default='steplr', help="学习率衰减器")
    parser.add_argument("--step_size", type=float, default=10, help="学习速率衰减的周期")
    parser.add_argument("--gamma", type=float, default=0.5, help="学习率衰减的乘法因子")

    #splitdata
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集地址")
    parser.add_argument("--data_path", type=str, default='/workspace/dataset/private', help="数据地址")
    parser.add_argument("--dataset_name", type=str, required=False, help="数据集名称")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集切割比例")
    parser.add_argument("--label_name", nargs="+", required=False, help="类别名称到数字映射")

    #TrainLoader or ValLoader
    parser.add_argument("--shuffle", type=str2bool, default='true', help="数据随机打乱")
    parser.add_argument("--batch_size", type=int, default=1, help="批次数")
    parser.add_argument("--num_workers", type=int, default=16, help="多少个进程处理数据")

    #train
    parser.add_argument("--pretrained_model", type=str, default=None, help="预训练模型")
    parser.add_argument("--save_model", type=str, default=None, help="模型保存路径")
    parser.add_argument("--epochs", type=int, required=False, help="训练的次数")
    parser.add_argument("--verbose", type=int, default=-1, help="训练多少次打印训练数据")
    parser.add_argument("--period", type=int, default=1 , help="训练多少次进行测试")


    '''==================================transplant==============================='''
    parser.add_argument("--weight", type=str, help ="模型路径")
    parser.add_argument("--quantized_mode", type=int, default=1, choices=[0, 1], help ="0-int8 1-int16")
    parser.add_argument('--quantized_dir', type=str, default='./quantized', help='量化模型保存地址')
    parser.add_argument('--offline_dir', type=str, default='./offline', help='离线模型保存地址')

    #quantize config
    parser.add_argument('--iteration', type=int, default=1, help='使用多少张图片量化')
    parser.add_argument("--use_avg", type=str2bool, default='false', help ="是否使用均值")
    parser.add_argument('--data_scale', type=float, default=1.0, help='图片缩放')
    parser.add_argument('--mean', nargs='+', default=[0,0,0], help='firstconv使用')
    parser.add_argument('--std', nargs='+', default=[1,1,1], help='firstconv使用')
    parser.add_argument('--per_channel', type=str2bool, default='false', help='通道量化')
    parser.add_argument('--firstconv', type=str2bool, default='false', help='使用firstconv')

    #offline
    parser.add_argument('--offline_batchsize', type=int, default=1, help='生成多少batchsize的离线模型')
    parser.add_argument('--core_number_270', type=int, default=16, help='生成多少核的离线模型')
    parser.add_argument('--core_number_220', type=int, default=4, help='生成多少核的离线模型')

    return parser.parse_args()