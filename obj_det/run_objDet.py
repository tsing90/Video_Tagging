import numpy as np
import cv2, os, datetime
from tqdm import tqdm
import argparse
import imageio
from collections import Counter

from torch.utils.data import Dataset, DataLoader
import torch
from detectron_api import obj_det_batch
from visual import draw_bbox
import detectron2.data.transforms as T


def get_best_labels(lab_one_sec, spl_rate, top_i=5):
    counter = Counter()
    i = 0
    for frame_lab in lab_one_sec:
        if spl_rate and i % spl_rate == 0:
            for label in frame_lab:
                counter[label] += 1

        i += 1

    top_lab = counter.most_common(top_i)  # list format
    ans = [tup[0] for tup in top_lab]
    return ans


def get_log(log_file, all_labels, fps, spl_rate=None):

    if os.path.isfile(log_file):
        os.remove(log_file)
    timestemp = 0
    with open(log_file, 'w') as log:
        for idx in range(len(all_labels)):
            if idx % fps == 0:

                txt = str(datetime.timedelta(seconds=timestemp))
                txt += '\t'
                
                if spl_rate:
                    best_labels = get_best_labels(all_labels[idx:idx+fps], spl_rate)
                else:
                    best_labels = all_labels[idx]
                for label in best_labels:
                    txt += label + ', '
                if len(best_labels) == 0:
                    txt += '\n'
                else:
                    txt = txt[:-2] + '\n'
                log.write(txt)
                timestemp += 1


class batch_imgs(Dataset):
    def __init__(self, img_dir, cfg):
        self.img_list = sorted(os.listdir(img_dir))
        self.dir_path = img_dir
        assert len(self.img_list) > 0, 'empty folder:%s' % img_dir
        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def __getitem__(self, index):
        img_path = os.path.join(self.dir_path, self.img_list[index])
        assert os.path.isfile(img_path)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        img_transf = self.transform_gen.get_transform(img).apply_image(img)
        img_tensor = torch.as_tensor(img_transf.astype("float32").transpose(2, 0, 1))
        img_inp = {"image": img_tensor, "height": height, "width": width}
        return img_inp, img

    def __len__(self):
        return len(self.img_list)


def collate_fn(batch):
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None, help='name of video file')
    parser.add_argument('--batch', default=4, type=int, help='the number of images as a batch')
    parser.add_argument('--threshold', default=0.6, type=float, help='the confidence of prediction')
    parser.add_argument('--demo', action='store_true', help='generate demo videos')
    args = parser.parse_args()

    vid_path = args.video
    assert os.path.isfile(vid_path), 'Error: wrong path of video: %s' % vid_path
    print('Processing video: %s' % vid_path)
    
    vid_dir = os.path.splitext(vid_path)[0]
    assert os.path.isdir(vid_dir)

    out_dir = os.path.splitext(vid_path)[0] + '_obj'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    predictor, all_class, cfg = obj_det_batch(threshold=args.threshold)  # threshold <<<----------------

    cap = cv2.VideoCapture(vid_path)
    flag, frame = cap.read()
    vid_fps = round(cap.get(cv2.CAP_PROP_FPS))
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = frame.shape[:2]

    scale_fps = vid_fps//24  # ATTENTION: target fps is around 24, thus reduce fps by `scale_fps`
    if scale_fps == 0:
        scale_fps = 1
    new_fps = vid_fps//scale_fps
    print('original resolution: {} x {}'.format(w,h))
    print('original fps: {}\nreduced to: {}'.format(vid_fps, new_fps))
    
    pbar = tqdm(total = 0)  # manual visualize
    pbar.total = vid_len // (scale_fps * args.batch)
    
    person_threshold = 0.9
    all_txt = []
    all_labels = []
    all_score = []
    #writer = imageio.get_writer(out_path, fps=new_fps)
    data_loader = DataLoader(dataset=batch_imgs(vid_dir, cfg), batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    img_list = sorted(os.listdir(vid_dir))
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i % scale_fps == 0:
                pbar.n = i // scale_fps
                pbar.refresh()

                frames = []
                imgs = []
                for t in data:
                    frames.append(t[0])
                    imgs.append(t[1])
                outputs = predictor(frames)
                for img, frame, output in zip(imgs, frames, outputs):
                    output = output["instances"]
                    pred_bbox = output.pred_boxes.tensor.detach().to('cpu').numpy()
                    pred_class = output.pred_classes.detach().to('cpu').numpy()
                    pred_score = output.scores.detach().to('cpu').numpy()

                    #txt_list = []
                    cls_stack = []
                    score_stack = []
                    for score, cls in zip(pred_score, pred_class):
                        label = all_class[cls]
                        if label not in cls_stack:
                            cls_stack.append(label)
                            score_stack.append(score)
                            #txt = '[{:.2f}] {}'.format(score, label)
                            #txt_list.append(txt)

                    # tracking purpose
                    person_bbox = pred_bbox[(pred_class == 0) & (pred_score > person_threshold)]
                    for bbox in person_bbox:
                        x1, y1, x2, y2 = bbox
                        w = x2 - x1
                        h = y2 - y1
                        txt = '{},{:.1f},{:.1f},{:.1f},{:.1f}\n'.format(i//scale_fps, x1, y1, w, h)
                        all_txt.append(txt)

                    # video purpose

                    new = draw_bbox(img, pred_bbox, pred_class, all_class)
                    #writer.append_data(new[:,:,::-1])
                    cv2.imwrite(os.path.join(out_dir, '{:05d}.jpg'.format(i // scale_fps)), new)
                    # data saving and log purpose
                    all_labels.append(cls_stack)
                    all_score.append(score_stack)

    
    pbar.close()
    # save video
    #writer.close()

    # write data into log file
    logAll_path = os.path.splitext(vid_path)[0] + '_objAll.log'
    get_log(logAll_path, all_labels, new_fps)  # timestamp based, use a single frame per second for logging

    """
    # save data
    all_labels = np.array(all_labels)
    all_score = np.array(all_score)
    npy_cls_path = os.path.splitext(vid_path)[0] + '_objCls.npy'
    npy_prob_path = os.path.splitext(vid_path)[0] + '_objProb.npy'
    np.save(npy_cls_path, all_labels)
    np.save(npy_prob_path, all_score)

    # write data into tracking_log file
    log_path = os.path.splitext(vid_path)[0]+'_objTrack.log'
    if os.path.isfile(log_path):
        os.remove(log_path)
    with open(log_path, 'w') as log:
        for txt in all_txt:
            log.write(txt)
    """
    print('log files are at:', logAll_path)
    if args.demo:
        obj_img_dir = os.path.splitext(vid_path)[0] + '_obj'
        os.system('ffmpeg -r {} -f image2 -i {} -vcodec libx264 {}'.format(vid_fps, obj_img_dir+"/%05d.jpg", obj_img_dir+'.mp4'))
    #shutil.rmtree(obj_img_dir)

