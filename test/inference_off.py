import argparse
from python_off import ModelOffline
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import torch

def main(opt):
    list_indx =opt.label_name
    img = Image.open(opt.input_img)
    
    input_size = opt.input_size
    if not isinstance(input_size, list) and not isinstance(input_size, tuple):
        input_size = (input_size, input_size)
    transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
    tf_image = transform(img).type(torch.FloatTensor)
    tf_image = np.transpose(tf_image[None,:],(0,2,3,1))
    tf_image = np.ascontiguousarray(tf_image)
    tf_image =np.array(tf_image,dtype =np.float32)
    # tf_image /=255.0
    model = ModelOffline(opt.off_model,0)
    result = model(tf_image)
    result_idx = np.argmax(result)
    print("test_result:",list_indx[result_idx])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='route_inference.py')
    parser.add_argument('--input_img', type=str, default='/workspace/volume/guojun/Train/Classification/dataset/test_cap/10.bmp', help='img not dir v1.0')
    parser.add_argument('--off_model', type=str, default='/workspace/volume/guojun/Train/Classification/offline/270cap_classification.cambricon', help='model_path')
    parser.add_argument("--input_size", type=int, default=100, help="model_size")
    parser.add_argument("--label_name", nargs="+", required=False, help="labelnum")
    opt = parser.parse_args()
    main(opt)
