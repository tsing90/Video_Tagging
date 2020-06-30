from __future__ import division
from torch.utils.data import Dataset, DataLoader
import getpass
import os, subprocess
import socket
import numpy as np
from .preprocess_data import *
from PIL import Image, ImageFilter
import pickle
import glob
# import dircache
import pdb


def get_test_video(opt, frame_path, Total_frames):
    """
        Args:
            opt         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : list of all video frames
        """

    clip = []
    i = 0
    loop = 0
    if Total_frames < opt.sample_duration: loop = 1

    if opt.modality == 'RGB':
        while len(clip) < max(opt.sample_duration, Total_frames):
            try:
                im = Image.open(os.path.join(frame_path, '%05d.jpg' % (i + 1)))
                clip.append(im.copy())
                im.close()
            except:
                pass
            i += 1

            if loop == 1 and i == Total_frames:
                i = 0

    elif opt.modality == 'Flow':
        while len(clip) < 2 * max(opt.sample_duration, Total_frames):
            try:
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg' % (i + 1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg' % (i + 1)))
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1

            if loop == 1 and i == Total_frames:
                i = 0

    elif opt.modality == 'RGB_Flow':
        while len(clip) < 3 * max(opt.sample_duration, Total_frames):
            try:
                im = Image.open(os.path.join(frame_path, '%05d.jpg' % (i + 1)))
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg' % (i + 1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg' % (i + 1)))
                clip.append(im.copy())
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im.close()
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1

            if loop == 1 and i == Total_frames:
                i = 0
    return clip


def get_train_video(opt, frame_path, Total_frames):
    """
        Chooses a random clip from a video for training/ validation
        Args:
            opt         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : random clip (list of frames of length sample_duration) from a video for training/ validation
        """
    clip = []
    i = 0
    loop = 0

    # choosing a random frame
    if Total_frames <= opt.sample_duration:
        loop = 1
        start_frame = np.random.randint(0, Total_frames)
    else:
        start_frame = np.random.randint(0, Total_frames - opt.sample_duration)

    if opt.modality == 'RGB':
        while len(clip) < opt.sample_duration:
            try:
                im = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i + 1)))
                clip.append(im.copy())
                im.close()
            except:
                pass
            i += 1

            if loop == 1 and i == Total_frames:
                i = 0

    elif opt.modality == 'Flow':
        while len(clip) < 2 * opt.sample_duration:
            try:
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg' % (start_frame + i + 1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg' % (start_frame + i + 1)))
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1

            if loop == 1 and i == Total_frames:
                i = 0

    elif opt.modality == 'RGB_Flow':
        while len(clip) < 3 * opt.sample_duration:
            try:
                im = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i + 1)))
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg' % (start_frame + i + 1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg' % (start_frame + i + 1)))
                clip.append(im.copy())
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im.close()
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1

            if loop == 1 and i == Total_frames:
                i = 0
    return clip


class HMDB51_test():
    """HMDB51 Dataset"""

    def __init__(self, vid_path, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : 0 for testing, 1 for training, 2 for validation
            split : 1,2,3
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_val_test = train
        self.opt = opt

        self.vid_path = vid_path
        assert os.path.isfile(self.vid_path)
        
        self.out_dir = os.path.join(os.path.splitext(self.vid_path)[0]+'_act_inp')
        if os.path.isdir(self.out_dir):
            print('the frames folder exists, done!')

        else:
            os.system('mkdir -p "%s"'%(self.out_dir))
            # check if horizontal or vertical scaling factor
            o = subprocess.check_output(
                'ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"' % (
                    self.vid_path), shell=True).decode('utf-8')
            lines = o.splitlines()
            width = int(lines[0].split('=')[1])
            height = int(lines[1].split('=')[1])
            resize_str = '-1:256' if width > height else '256:-1'

            # extract frames
            os.system('ffmpeg -i "%s" -r "%s" -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1' % (
            self.vid_path, opt.fps, resize_str, os.path.join(self.out_dir, '%05d.jpg')))

        self.Total_frames = len([fname for fname in os.listdir(self.out_dir) if fname.endswith('.jpg') and len(fname) == 9])
        if self.Total_frames == 0: raise Exception


    def __len__(self):
        '''
        returns number of test/train set
        '''
        return self.Total_frames

    def get_clip(self):

        clip = []
        i = 0
        loop = 0

        Total_frames = self.Total_frames

        if Total_frames < self.opt.sample_duration: loop = 1

        if self.opt.modality == 'RGB':
            while len(clip) < max(self.opt.sample_duration, Total_frames):
                try:
                    im = Image.open(os.path.join(self.out_dir, '%05d.jpg' % (i + 1)))
                    clip.append(im.copy())
                    im.close()
                except:
                    print('Error: %05d.jpg'%(i+1))
                i += 1

                if loop == 1 and i == Total_frames:
                    i = 0

        return scale_crop(clip, self.train_val_test, self.opt)


class UCF101_test(Dataset):
    """UCF101 Dataset"""

    def __init__(self, vid_path, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : 0 for testing, 1 for training, 2 for validation
            split : 1,2,3
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_val_test = train
        self.opt = opt

        self.vid_path = vid_path
        assert os.path.isfile(self.vid_path)
        self.out_dir = os.path.join(self.vid_path[:-4])
        if os.path.isdir(self.out_dir):
            print('the frames folder exists, done!')

        else:
            os.system('mkdir -p "%s"'%(self.out_dir))
            # check if horizontal or vertical scaling factor
            o = subprocess.check_output(
                'ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"' % (
                    self.vid_path), shell=True).decode('utf-8')
            lines = o.splitlines()
            width = int(lines[0].split('=')[1])
            height = int(lines[1].split('=')[1])
            resize_str = '-1:256' if width > height else '256:-1'

            # extract frames
            os.system('ffmpeg -i "%s" -r 25 -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1' % (
            self.vid_path, resize_str, os.path.join(self.out_dir, '%05d.jpg')))

        self.Total_frames = len([fname for fname in os.listdir(self.out_dir) if fname.endswith('.jpg') and len(fname) == 9])
        if self.Total_frames == 0: raise Exception


    def __len__(self):
        '''
        returns number of test/train set
        '''
        return self.Total_frames

    def get_clip(self):

        clip = []
        i = 0
        loop = 0

        Total_frames = self.Total_frames

        if Total_frames < self.opt.sample_duration: loop = 1

        if self.opt.modality == 'RGB':
            while len(clip) < max(self.opt.sample_duration, Total_frames):
                try:
                    im = Image.open(os.path.join(self.out_dir, '%05d.jpg' % (i + 1)))
                    clip.append(im.copy())
                    im.close()
                except:
                    print('Error: %05d.jpg'%(i+1))
                i += 1

                if loop == 1 and i == Total_frames:
                    i = 0

        return scale_crop(clip, self.train_val_test, self.opt), clip


class Kinetics_test(Dataset):
    def __init__(self, split, train, opt):
        """
        Args:
            opt   : config options
            train : 0 for testing, 1 for training, 2 for validation
            split : 'val' or 'train'
        Returns:
            (tensor(frames), class_id ) : Shape of tensor C x T x H x W
        """
        self.split = split
        self.opt = opt
        self.train_val_test = train

        # joing labnames with underscores
        self.lab_names = sorted([f for f in os.listdir(os.path.join(self.opt.frame_dir, "train"))])

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 400

        # indexes for validation set
        if train == 1:
            label_file = os.path.join(self.opt.annotation_path, 'Kinetics_train_labels.txt')
        else:
            label_file = os.path.join(self.opt.annotation_path, 'Kinetics_val_labels.txt')

        self.data = []  # (filename , lab_id)

        f = open(label_file, 'r')
        for line in f:
            class_id = int(line.strip('\n').split(' ')[-2])
            nb_frames = int(line.strip('\n').split(' ')[-1])
            self.data.append(
                (os.path.join(self.opt.frame_dir, ' '.join(line.strip('\n').split(' ')[:-2])), class_id, nb_frames))
        f.close()

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = video[1]
        frame_path = video[0]
        Total_frames = video[2]

        if self.opt.only_RGB:
            Total_frames = len(glob.glob(glob.escape(frame_path) + '/0*.jpg'))
        else:
            Total_frames = len(glob.glob(glob.escape(frame_path) + '/TVL1jpg_y_*.jpg'))

        if self.train_val_test == 0:
            clip = get_test_video(self.opt, frame_path, Total_frames)
        else:
            clip = get_train_video(self.opt, frame_path, Total_frames)

        return ((scale_crop(clip, self.train_val_test, self.opt), label_id))


