from dataset.new_dataset import *
from tqdm import tqdm
import cv2
import os
import numpy as np
import imageio
from video_process import txt2img

import getpass
import socket
from dataset.preprocess_data import *
from PIL import Image, ImageFilter
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from models.model import generate_model
from opts import parse_opts
from torch.autograd import Variable
import time
import torch.utils
import sys
from utils import *
import pdb
    
def test_hmdb(vid_path,vid_out):

    # print configuration options
    opt = parse_opts()
    #print(opt)

    #vid_path = os.path.join('../Videos/data/Clapping', 'PL_19_20_ARS_WHU-5492.ts')
    assert os.path.isfile(vid_path), 'wrong path:{}'.format(vid_path)
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if num_f == 0:
        print('no frames in the video')
        return 
    
    label_file = os.path.join('dataset', 'HMDB51_labels.txt')
    assert os.path.isfile(label_file)
    all_cls = []
    with open(label_file, 'r') as f:
        for i in range(opt.n_classes):
            all_cls.append(f.readline().strip())
    #print(all_cls)
    all_cls = np.array(all_cls)

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    print("Preprocessing video ...")
    data = HMDB51_test(vid_path, split = opt.split, train = 0, opt = opt)
    print("Frames of video: ", len(data))
    
    if opt.modality=='RGB': opt.input_channels = 3
    elif opt.modality=='Flow': opt.input_channels = 2
    
    # Loading model and checkpoint
    model, parameters = generate_model(opt)
    if opt.resume_path1:
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        assert opt.arch == checkpoint['arch']
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
        
    with torch.no_grad():
        clip, origin_clip = data.get_clip()
        clip = torch.squeeze(clip)

        num_sample = int(clip.shape[1]/opt.sample_duration)
        inputs = torch.Tensor(1, 3, opt.sample_duration, opt.sample_size, opt.sample_size)
        top5_out = []
        sfmx = nn.Softmax(dim=1)
        for k in tqdm(range(num_sample)):
            inputs[0,:,:,:,:] = clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:]
            inputs_var = Variable(inputs)
            outputs_var= model(inputs_var)
            outputs_var= sfmx(outputs_var) 

            pred5 = outputs_var.topk(5, 1, True)
            pred5_str = all_cls[pred5[1].cpu().data[0].numpy()]
            pred5_prob = pred5[0].cpu().data[0].numpy()

            top5_out.append((pred5_prob, pred5_str))

        mod = clip.shape[1] % opt.sample_duration
        if mod != 0:
            inputs[0, :, :, :, :] = clip[:, -opt.sample_duration:, :, :]
            inputs_var = Variable(inputs)
            outputs_var= model(inputs_var)
            outputs_var= sfmx(outputs_var)

            pred5 = outputs_var.topk(5, 1, True)
            pred5_str = all_cls[pred5[1].cpu().data[0].numpy()]
            pred5_prob = pred5[0].cpu().data[0].numpy()

            top5_out.append((pred5_prob, pred5_str))

    #vid_out = vid_path[:-4]+'_act.mp4'
    assert os.path.isdir(os.path.dirname(vid_out))
    print('making a video ...')
    with imageio.get_writer(vid_out, fps=fps) as writer:
        imageio.plugins.ffmpeg.download()
        for i, img in enumerate(tqdm(origin_clip)):
            img = np.array(img)
            pred5_i_prob, pred5_i_str = top5_out[i // opt.sample_duration]
            txt = []
            for i in range(5):
                txt_j = '[{:.2f}] {}'.format(pred5_i_prob[i], pred5_i_str[i])
                txt.append(txt_j)

            new = txt2img(img, txt, font_scale=0.4, y_offset=25, y_margin=15, x_offset=10)
            writer.append_data(new)
    writer.close()

