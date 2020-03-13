import os, sys
from detectron2.config import get_cfg
from detectron2.engine import default_setup
import logging
import numpy as np
import cv2
from detectron2.engine import DefaultPredictor
import torch
import math
from tools.birdview_detection_refiner import BirdviewDetectionRefiner
from tools.utils_3d import _draw_projection_obstacle_to_cam
from tools.object_3d import Object3d
from tools.utils_calib import Calibration
import os
detectron2_root = os.getenv('DETECTRON_ROOT')

config_path = os.path.join(detectron2_root, 'configs', 'Base-BirdNetPlus.yaml')
checkpoint_path = os.path.join(detectron2_root, 'models', 'model_ITSC2020.pth')
BEV_im_path = os.path.join(detectron2_root, 'demo', 'demo_img', '000021_bev.png')
im_path = os.path.join(detectron2_root, 'demo', 'demo_img', '000021.png')
calib_path = os.path.join(detectron2_root, 'demo', 'demo_img', '000021.txt')

idclass3 = { 0:'Car', 1:'Pedestrian', 2:'Cyclist'}

def prepareAnn(lbl, alpha, box, h=-1, w=-1, l=-1, x=-1000, y=-1000, z=-1000, ry=-10, score=None):
    ann = [
       lbl,
       -1,
       -1,
       alpha,
       box[0],box[1],box[2],box[3],
       h,w,l,
       x,y,z,
       ry
    ]
    if score is not None:
        ann.append(score)
    strAnn = ' '.join([str(x) for x in ann])
    obj3d = Object3d(strAnn)

    return ann, obj3d, strAnn

def prepare_for_coco_detection_KITTI(instance, output_folder, filename, write, calib_file, vp, bins, vp_res, hwrot, height_training):
    # Extract important information from instance class
    boxes  = np.array(instance.get('pred_boxes').tensor)
    scores = np.array(instance.get('scores'))
    labels = np.array(instance.get('pred_classes'))
    alpha = np.array([rad for rad in instance.get('viewpoint_residual')]) if vp else np.ones((labels.shape))*(-10.00)
    h = np.array([[h,g] for h,g in instance.get('height')]) if height_training else np.array([-1,-1000]*labels.shape)

    # Image BV
    bv_image = cv2.imread(filename).astype(np.uint8)
    bv_ground = None

    # Refiner for 3D
    refiner = BirdviewDetectionRefiner(bv_image, bv_ground, 0.05, 1.73, True)

    im_ann = []
    im_ann_obj = []
    for k, box in enumerate(boxes):
        lbl = idclass3[labels[k]]
        ann,obj3d,strAnn = prepareAnn(lbl,alpha[k],box,score=scores[k],h=h[k,0],z=h[k,1])
        refiner.refine_detection_rotated_wheight(obj3d)
        if obj3d.height == -1:
            continue

        # Project points to camera frame coordinates
        calib = Calibration(calib_file)
        p = calib.project_velo_to_rect(np.array([[obj3d.location.x,obj3d.location.y,obj3d.location.z]]))

        # Change 2D bbox in BV getting 2D bbox in camera frame (projection)
        _,_,bbox2D = _draw_projection_obstacle_to_cam(obj3d, calib_file, 0.05, True, False)
        if bbox2D == None:
            continue
        # Obtain alpha from yaw
        obj3d.alpha = obj3d.yaw -(-math.atan2(p[0][2],p[0][0]) - 1.5*math.pi)
        obj3d.alpha = obj3d.alpha%(2*math.pi)
        if obj3d.alpha > math.pi:
            obj3d.alpha -= 2*math.pi
        elif obj3d.alpha < -math.pi:
            obj3d.alpha += 2*math.pi

        # After refinement
        ann = [
               obj3d.kind_name,
               obj3d.truncated,
               obj3d.occluded,
               round(obj3d.alpha,6),
               round(bbox2D[0],6),round(bbox2D[1],6),round(bbox2D[2],6),round(bbox2D[3],6),
               round(obj3d.height,6), round(obj3d.width,6), round(obj3d.length,6),
               round(p[0][0],6), round(p[0][1],6), round(p[0][2],6), # Camera coordinates
               round(obj3d.yaw,6),
               obj3d.score, # DON'T ROUND IT
            ]

        im_ann.append(ann)
        im_ann_obj.append(obj3d)
        strAnn = ' '.join([str(x) for x in ann])

        if write:
            file_ann.write(strAnn+'\n')
    if write:
        file_ann.close()
    return  im_ann, im_ann_obj, instance

def main():
    logger = logging.getLogger("detectron2.trainer")
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    default_setup(cfg, None)
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    predictor = DefaultPredictor(cfg)

    im = cv2.imread(BEV_im_path)
    outputs = predictor(im)
    _, obj_anns, _ = prepare_for_coco_detection_KITTI(outputs["instances"].to("cpu"), cfg.OUTPUT_DIR, BEV_im_path, False, calib_path, cfg.VIEWPOINT, cfg.VP_BINS, cfg.VIEWPOINT_RESIDUAL, cfg.ROTATED_BOX_TRAINING, cfg.HEIGHT_TRAINING)
    # Show 2D and 3D images
    # Change BV aspect
    nonzero = np.where(im>0)
    im[nonzero]=255-im[nonzero]
    im=cv2.bitwise_not(im)

    kitti_im = cv2.imread(im_path)
    # Show obstacles
    for obj in obj_anns:
        kitti_im, im, _ = _draw_projection_obstacle_to_cam(obj, calib_path, 0.05, True, True, kitti_im, im)
    cv2.imshow('image',kitti_im)
    cv2.imshow('bv_image',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == '__main__':
    main()