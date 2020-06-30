import imageio
from imageio import get_writer
from tqdm import tqdm
import os, cv2

def vid2img(vid_path, out_dir):
    cap = cv2.VideoCapture(vid_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    flag, img = cap.read()
    i = 0

    print('start ...')
    while flag:
        path = os.path.join(out_dir, '{:05d}.jpg'.format(i))
        cv2.imwrite(path, img)
        i += 1
        flag, img = cap.read()
        
    print('end !')
    return fps
    
def img2vid(out_dir, fps):
    imageio.plugins.ffmpeg.download()
    
    with get_writer('bmg_1_scene.mp4', fps=fps) as writer:
        for name in tqdm(sorted(os.listdir(out_dir))):
            img_path = os.path.join(out_dir, name)
            img = cv2.imread(img_path)
            writer.append_data(img)
            
    writer.close()
    
def txt2img(img, txt_list, font_scale=1, y_offset=50, y_margin=30, x_offset=10):
    # txt_list: must be a list 
    # common text: '[{:.2f}] {}'.format(probs[i], classes[idx[i]])

    for txt in txt_list:
        cv2.putText(img, txt, (x_offset, y_offset), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0, 235, 0), thickness=2)
        y_offset += y_margin
    
    return img
    
