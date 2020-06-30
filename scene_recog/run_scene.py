
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os, cv2, time
from PIL import Image
from tqdm import tqdm
import argparse
from collections import Counter
import imageio
import datetime


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
                    best_labels = get_best_labels(all_labels[idx:idx + fps], spl_rate, top_i=3)
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


parser = argparse.ArgumentParser()
parser.add_argument('--video', default=None, help='name of video file')
parser.add_argument('--demo', default=False, help='generate demo videos')
args = parser.parse_args()

# load the test image
vid_path = args.video
img_dir = os.path.splitext(vid_path)[0]  # this is expected to be an image folder!
assert os.path.isdir(img_dir), 'directory not existing:{}'.format(img_dir)
#out_dir = img_dir+'_scene_out'
#if not os.path.isdir(out_dir):
#    os.mkdir(out_dir)
cap = cv2.VideoCapture(vid_path)
fps = round(cap.get(cv2.CAP_PROP_FPS))

# th architecture to use
arch = 'resnet50'

# load the pre-trained weights
model_dir="model"
model_name = '%s_places365.pth.tar' % arch
model_file = os.path.join(model_dir, model_name)

model = models.__dict__[arch](num_classes=365)
if torch.cuda.is_available():
    print('use GPU!')
    device = torch.device('cuda')
    checkpoint = torch.load(model_file)  # gpu usage
else:
    print('using cpu')
    device = torch.device('cpu')
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)

state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

if args.demo:
    writer = imageio.get_writer(os.path.splitext(vid_path)[0]+'_scene.mp4', fps=fps)
all_labels = []
threshold = 0.05  # <<<<<<<<<<<<<<<------------------------
for name in tqdm(sorted(os.listdir(img_dir))):
    img_name = os.path.join(img_dir, name)
    
    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))
    input_img = input_img.to(device)

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    img = cv2.imread(img_name)
    y_offset = 50
    cls_stack = []
    for i in range(3):
        if probs[i] > threshold:
            cls_stack.append(classes[idx[i]])
            cv2.putText(img, '[{:.2f}] {}'.format(probs[i], classes[idx[i]]), (10, y_offset),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 235, 0), thickness=2)
            y_offset += 40

    if args.demo:
        writer.append_data(img[:, :, ::-1])
    all_labels.append(cls_stack)
    
# save video
if args.demo:
    writer.close()
# write data into log file
log_path = os.path.splitext(vid_path)[0] + '_scene.log'
get_log(log_path, all_labels, fps, spl_rate=2)  # spl_rate is not important
