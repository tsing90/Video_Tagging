import numpy as np
import cv2, os
from tqdm import tqdm

# from detectron_api import obj_det
from video_process import txt2img
# from visual import draw_bbox, random_color

import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T


class batch_imgs(Dataset):
    def __init__(self, img_dir):
        self.img_list = sorted(os.listdir(img_dir))
        self.dir_path = img_dir
        assert len(self.img_list) > 0, 'empty folder:%s' % img_dir
        """self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )"""

    def __getitem__(self, index):
        img_path = os.path.join(self.dir_path, self.img_list[index])
        assert os.path.isfile(img_path)
        img = cv2.imread(img_path)
        # img = self.transform_gen.get_transform(img).apply_image(img)
        img_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        img_inp = {'image': img_tensor}
        return img_inp

    def __len__(self):
        return len(self.img_list)


def collate_fn(batch):
    return batch


if __name__ == '__main__':
    # predictor, all_class = obj_det(threshold=0.1)

    cfg = get_cfg()
    cfg_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

    cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    model = build_model(cfg)
    DetectionCheckpointer(model).load('/home/ubuntu/.torch/fvcore_cache/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl')
    cfg.MODEL.WEIGHTS = "..."
    # model = DataParallel(model, device_ids=[0, 1])
    model.train(False)

    all_class = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get('thing_classes')

    img_dir = '../data/stringer_1/lake'
    assert os.path.isdir(img_dir)

    data_loader = DataLoader(dataset=batch_imgs(img_dir), batch_size=4, shuffle=False, collate_fn=collate_fn)
    # data_loader = build_batch_data_loader()

    for data in data_loader:
        outputs = model(data)
        print('one batch output got!')
        print(outputs.shape)
    print('done')


    def draw(img, img_path, output):
        output = output["instances"].to('cpu')
        pred_class = output.pred_classes.numpy()
        pred_score = output.scores.numpy()

        txt_list = []
        for score, cls in zip(pred_score, pred_class):
            txt = '[{:.2f}] {}'.format(score, all_class[cls])
            txt_list.append(txt)

        new = txt2img(img, txt_list)
        cv2.imwrite(os.path.splitext(img_path)[0] + '_obj.jpg', new)
