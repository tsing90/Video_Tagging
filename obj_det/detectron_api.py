
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def obj_det(cfg_name=None, threshold=0.5):
    cfg = get_cfg()
    if not cfg_name:
        # frnn_x_101_fpn has highest performance, but rather slow; 
        #cfg_name = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        #cfg_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        cfg_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

    cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)
    predictor = DefaultPredictor(cfg)
    all_class = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get('thing_classes')
    
    return predictor, all_class


def obj_det_batch(cfg_name=None, threshold=0.5):
    cfg = get_cfg()
    if not cfg_name:
        cfg_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

    cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
    model = build_model(cfg)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)
    #DetectionCheckpointer(model).load('/home/ubuntu/.torch/fvcore_cache/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl')
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    # model = DataParallel(model, device_ids=[0, 1])
    model.eval()

    all_class = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get('thing_classes')

    return model, all_class, cfg
