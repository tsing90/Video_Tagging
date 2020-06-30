import numpy as np
import cv2, os, datetime
from tqdm import tqdm
import argparse
import imageio
from collections import Counter

from detectron_api import obj_det
from visual import draw_bbox

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
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None, help='name of video file')
    args = parser.parse_args()
    
    #vid_name = '21js_1.mov'
    vid_name = args.video
    print('Processing video: %s' % vid_name)
    
    #vid_dir = 'data/vid'
    #out_dir = 'data/vid_out'
    #assert os.path.isdir(vid_dir)
    #assert os.path.isdir(out_dir)
    vid_path = vid_name

    #out_path = os.path.splitext(vid_path)[0]+'_obj.mp4'
    out_dir = os.path.splitext(vid_path)[0] + '_obj'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    predictor, all_class = obj_det(threshold=0.6)  # threshold <<<----------------
    
    cap = cv2.VideoCapture(vid_path)
    flag, frame = cap.read()
    vid_fps = round(cap.get(cv2.CAP_PROP_FPS))
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = frame.shape[:2]
    
    i = 0
    scale_fps = vid_fps//24  # ATTENTION: target fps is around 24, thus reduce fps by `scale_fps`
    if h > 1100:
        scale_res = h // 1080  # ATTENTION: reduce resolution by `scale_res`
    else:
        scale_res = 1
    
    new_fps = vid_fps//scale_fps
    new_h, new_w = h//scale_res, w//scale_res
    print('original resolution: {} x {}\nreduced to:{} x {}'.format(w,h,new_w,new_h))
    print('original fps: {}\nreduced to: {}'.format(vid_fps, new_fps))
    
    pbar = tqdm(total = 0)  # manual visualize
    pbar.total = vid_len // scale_fps
    
    person_threshold = 0.9
    all_txt = []
    all_labels = []
    all_score = []
    #writer = imageio.get_writer(out_path, fps=new_fps)
    while flag:
        i += 1
        
        if i % scale_fps == 0:
            pbar.n = i // scale_fps
            pbar.refresh()
            
            # down-size frame
            frame = cv2.resize(frame, (new_w, new_h))
            
            output = predictor(frame)["instances"].to('cpu')
            pred_bbox = output.pred_boxes.tensor.numpy()
            pred_class = output.pred_classes.numpy()
            pred_score = output.scores.numpy()
            
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
            new = draw_bbox(frame, pred_bbox, pred_class, all_class)
            #writer.append_data(new[:,:,::-1])
            cv2.imwrite(os.path.join(out_dir, '{:05d}.jpg'.format(i // scale_fps)), new)
            # data saving and log purpose
            all_labels.append(cls_stack)
            all_score.append(score_stack)
            
        flag, frame = cap.read()
    
    pbar.close()
    # save video
    #writer.close()

    # write data into log file
    logAll_path = os.path.splitext(vid_path)[0] + '_objAll.log'
    get_log(logAll_path, all_labels, new_fps)  # timestamp based, use a single frame per second for logging

    # save data
    all_labels = np.array(all_labels)
    all_score = np.array(all_score)
    npy_cls_path = os.path.splitext(vid_path)[0] + '_objCls.npy'
    npy_prob_path = os.path.splitext(vid_path)[0] + '_objProb.npy'
    np.save(npy_cls_path, all_labels)
    np.save(npy_prob_path, all_score)

    # write data into tracking_log file
    log_path = os.path.splitext(vid_name)[0]+'_objTrack.log'
    if os.path.isfile(log_path):
        os.remove(log_path)
    with open(log_path, 'w') as log:
        for txt in all_txt:
            log.write(txt)
    
    print('log files are at:', log_path)
