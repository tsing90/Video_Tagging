import argparse
from tqdm import tqdm
from collections import Counter
import datetime

from utils import google_utils
from utils.datasets import *
from utils.utils import *


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=None, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=None, help='output folder')  # output folder
    parser.add_argument('--demo', action='store_true', help='produce a video & images for demo')
    parser.add_argument('--trackingLog', action='store_true', help='produce log file for Deniz tracking algo')

    parser.add_argument('--weights', type=str, default='weights/yolov5l.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()

    assert opt.video, 'Error: wrong video path: %s' % opt.video
    assert opt.weights, 'Error: wrong model weight path: %s ' % opt.weights
    if not opt.output:
        pass
    opt.img_size = check_img_size(opt.img_size)
    return opt


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
                    best_labels = get_best_labels(all_labels[idx:idx + fps], spl_rate)
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


def obj_detect(opt):
    out, source, weights, imgsz = \
        opt.output, opt.video, opt.weights, opt.img_size

    # Initialize
    device = torch_utils.select_device(opt.device)
    half = False  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    # Cause issue when running out of yolo folder !!!
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path = opt.video
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    all_txt = []
    all_labels = []
    all_score = []
    person_threshold = 0.9

    scale_fps = dataset.fps//24  # ATTENTION: target fps is around 24, thus reduce fps by `scale_fps`
    if scale_fps == 0:
        scale_fps = 1
    new_fps = dataset.fps//scale_fps
    print('original fps: {}\nreduced to: {}'.format(dataset.fps, new_fps))
    pbar = tqdm(total = 0)  # manual visualize
    pbar.total = dataset.nframes // scale_fps
    if opt.demo:
        vid_writer = cv2.VideoWriter(os.path.splitext(vid_path)[0]+'_obj.mp4', cv2.VideoWriter_fourcc(*opt.fourcc),
                                     dataset.fps, (dataset.width, dataset.height))
        out_dir = os.path.splitext(vid_path)[0] + '_obj'
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

    for path, img, im0, vid_cap in dataset:
        pbar.n = dataset.frame // scale_fps
        pbar.refresh()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        cls_stack = []
        score_stack = []
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in det:
                    # tracking purpose
                    if opt.trackingLog and int(cls) == 0 and conf > person_threshold:
                        x1, y1, x2, y2 = xyxy
                        w = x2 - x1
                        h = y2 - y1
                        txt = '{},{:.1f},{:.1f},{:.1f},{:.1f}\n'.format(i // scale_fps, x1, y1, w, h)
                        all_txt.append(txt)

                    label = names[int(cls)]
                    if label not in cls_stack:
                        cls_stack.append(label)
                        score_stack.append(conf)

                    if opt.demo:
                        label_plot = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label_plot, color=colors[int(cls)], line_thickness=3)

        # log & debug purpose
        all_labels.append(cls_stack)
        all_score.append(score_stack)
        # demo purpose
        if opt.demo:
            cv2.imwrite(os.path.join(out_dir, '{:05d}.jpg'.format(dataset.frame // scale_fps)), im0)
            vid_writer.write(im0)

    pbar.close()
    # save video
    # writer.close()

    # write data into log file
    logAll_path = os.path.splitext(vid_path)[0] + '_objAll.log'
    get_log(logAll_path, all_labels, new_fps)  # timestamp based, use a single frame per second for logging
    print('log files are at:', logAll_path)

    # write data into tracking_log file
    if opt.trackingLog:
        log_path = os.path.splitext(vid_path)[0] + '_objTrack.log'
        if os.path.isfile(log_path):
            os.remove(log_path)
        with open(log_path, 'w') as log:
            for txt in all_txt:
                log.write(txt)

    # shutil.rmtree(obj_img_dir)
    """
    # debug only 
    # save data
    all_labels = np.array(all_labels)
    all_score = np.array(all_score)
    npy_cls_path = os.path.splitext(vid_path)[0] + '_objCls.npy'
    npy_prob_path = os.path.splitext(vid_path)[0] + '_objProb.npy'
    np.save(npy_cls_path, all_labels)
    np.save(npy_prob_path, all_score)
    """


if __name__ == '__main__':
    opt = get_opt()
    with torch.no_grad():
        obj_detect(opt)
