import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from network import *
import time

parser = argparse.ArgumentParser(description="RDARENet_Test")
parser.add_argument("--model_dir", type=str, default="models/Rain100H.pth", help='path to model and log files')
parser.add_argument("--img_path", type=str, default="data/input/rain100H.png", help='path to testing data')
parser.add_argument("--save_path", type=str, default="results", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')

opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = RDARENet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_GPU)

    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(opt.model_dir))
    model.eval()



    img_path = opt.img_path
    img_name = opt.img_path.split('/')[-1]

    # input image
    y = cv2.imread(img_path)
    b, g, r = cv2.split(y)
    y = cv2.merge([r, g, b])

    y = normalize(np.float32(y))
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    y = Variable(torch.Tensor(y))

    if opt.use_GPU:
        y = y.cuda()

    with torch.no_grad(): #
        if opt.use_GPU:
            torch.cuda.synchronize()
        start_time = time.time()

        out, _ = model(y)
        out = torch.clamp(out, 0., 1.)

        if opt.use_GPU:
            torch.cuda.synchronize()

    if opt.use_GPU:
        save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
    else:
        save_out = np.uint8(255 * out.data.numpy().squeeze())

    save_out = save_out.transpose(1, 2, 0)
    b, g, r = cv2.split(save_out)
    save_out = cv2.merge([r, g, b])

    cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)


if __name__ == "__main__":
    main()

