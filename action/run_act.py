from dataset.new_dataset import *
from tqdm import tqdm
import cv2
import os
import numpy as np
import imageio
import shutil
from video_process import txt2img

from torch import nn
from models.model import generate_model
from opts import parse_opts
from torch.autograd import Variable
import datetime
import torch.utils
import sys
from utils import *
    
if __name__=="__main__":
    # print configuration options
    opt = parse_opts()
    #print(opt)

    #vid_path = os.path.join('../data', '21js_1.mov')
    # three dir param: video , dir_input, dir_output
    vid_path = os.path.join(opt.video)
    assert os.path.isfile(vid_path), 'wrong path:{}'.format(vid_path)
  
    cap = cv2.VideoCapture(vid_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    opt.fps = fps
    
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

    threshold = 0.2
    with torch.no_grad():
        clip = data.get_clip()
        clip = torch.squeeze(clip)

        num_sample = int(clip.shape[1]/opt.sample_duration)
        inputs = torch.Tensor(1, 3, opt.sample_duration, opt.sample_size, opt.sample_size)
        top5_out = []  # changed to top-2
        sfmx = nn.Softmax(dim=1)
        for k in tqdm(range(num_sample)):
            inputs[0,:,:,:,:] = clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:] # 1xCx64xHxW
            inputs_var = Variable(inputs) # 1xCx64xHxW
            outputs_var= model(inputs_var)  # 2048 x 64
            outputs_var= sfmx(outputs_var) # 1x51

            pred5 = outputs_var.topk(2, 1, True)
            pred5_str = all_cls[pred5[1].cpu().data[0].numpy()]
            pred5_prob = pred5[0].cpu().data[0].numpy()

            pred5_str = pred5_str[pred5_prob > threshold]
            pred5_prob = pred5_prob[pred5_prob > threshold]
            top5_out.append((pred5_prob, pred5_str))

        mod = clip.shape[1] % opt.sample_duration
        if mod != 0:
            inputs[0, :, :, :, :] = clip[:, -opt.sample_duration:, :, :]
            inputs_var = Variable(inputs)
            outputs_var= model(inputs_var)
            outputs_var= sfmx(outputs_var)

            pred5 = outputs_var.topk(2, 1, True)
            pred5_str = all_cls[pred5[1].cpu().data[0].numpy()]
            pred5_prob = pred5[0].cpu().data[0].numpy()

            pred5_str = pred5_str[pred5_prob > threshold]
            pred5_prob = pred5_prob[pred5_prob > threshold]
            top5_out.append((pred5_prob, pred5_str))

    # making a log
    log_path = os.path.splitext(vid_path)[0] + '_act.log'
    with open(log_path, 'w') as log:
        for i in range(len(data)):
            if i % fps == 0:
                txt = str(datetime.timedelta(seconds=i//fps))
                txt += '\t'

                _, pred_str = top5_out[i//opt.sample_duration]
                for label in pred_str:
                    txt += label + ', '
                if len(pred_str) == 0:
                    txt += '\n'
                else:
                    txt = txt[:-2] + '\n'
                log.write(txt)

    # making video
    if opt.demo:
        vid_out = os.path.splitext(vid_path)[0]+'_act.mp4'
        print('making a video ...')
        flag, frame = cap.read()
        with imageio.get_writer(vid_out, fps=fps) as writer:
            #imageio.plugins.ffmpeg.download()
            t = 0
            while flag:
                pred5_i_prob, pred5_i_str = top5_out[t // opt.sample_duration]
                txt = []
                for i in range(len(pred5_i_str)):
                    txt_j = '[{:.2f}] {}'.format(pred5_i_prob[i], pred5_i_str[i])
                    txt.append(txt_j)

                new = txt2img(frame, txt, font_scale=0.8, y_offset=50, y_margin=30, x_offset=10)
                writer.append_data(new[:, :, ::-1])
                flag, frame = cap.read()
                t += 1
        writer.close()

    shutil.rmtree(os.path.splitext(vid_path)[0] + '_act_inp')

