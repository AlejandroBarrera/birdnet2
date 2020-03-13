#!/usr/local/bin/python3
import os, sys
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg 
from detectron2.engine import DefaultTrainer, default_setup
import logging
import torch
import tools.convert_kitti_to_coco_rotation as gen
import argparse
# Paths
detectron2_root = os.getenv('DETECTRON_ROOT')

'''
The algorithm makes possible to train with various use cases:
1. Only viewpoint, as it was with the paper BirdNet, but using detectron2 as platform and new backbones
2. New box encoding to train 3D boxes, although it continues estimating height in post-processing
3. At the same level, it is possible to activate orientation refinement as well as height and elevation estimate
All the annotations are created from this script just reading from config files in home/detectron2/configs
Enjoy!
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for BirdNet+')
    parser.add_argument(
        '--config_file', help="Name of the configuration to use without extension", default='Base-BirdNetPlus', type=str)
    parser.add_argument(
        '--training_classes', help="Index of the classes to train with, corresponding to 'Car:0', 'Van:1', 'Truck:2', 'Pedestrian:3', 'Person_sitting:4', 'Cyclist:5', 'Tram:6', 'Misc:7', 'DontCare:8'",\
         default='0,3,5', type=str)
    return parser.parse_args()

def main(config_file, training_classes):
    # Logger and configuration load
    logger = logging.getLogger("detectron2.trainer")
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(detectron2_root,"configs/{}.yaml".format(config_file)))
    default_setup(cfg, None)

    # Registering datasets and different fields, it must be configured in yaml file
    nclasses = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    ann_data_dir = detectron2_root+'/datasets/bv_kitti'
    ann_out_dir = detectron2_root+'/datasets/bv_kitti/annotations'
    ann_val_file = detectron2_root+'/datasets/bv_kitti/lists/valsplit_chen.txt'
    ann_train_file = detectron2_root+'/datasets/bv_kitti/lists/trainsplit_chen.txt'
    ann_bins = int(cfg.VP_BINS)

    optional_arguments = []
    if cfg.VIEWPOINT:
        optional_arguments.append('viewpoint')
    else:
        cfg.ROTATED_BOX_TRAINING = False 
    if cfg.VIEWPOINT_RESIDUAL:
        optional_arguments.append('vp_res')
    if cfg.ROTATED_BOX_TRAINING:
        optional_arguments.append('bbox3D')
    if cfg.HEIGHT_TRAINING:
        optional_arguments.append('height')

    # Generate annotations
    train_path = gen.convert_kitti_training(ann_data_dir, ann_out_dir, ann_val_file, ann_train_file,\
                training_classes, ann_bins, cfg.VIEWPOINT, cfg.VIEWPOINT_RESIDUAL, cfg.ROTATED_BOX_TRAINING, cfg.HEIGHT_TRAINING)
    register_coco_instances("birdview_train", {}, train_path, detectron2_root, optional_arguments)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 

    # TRAINING MODEL
    # Set False here to take a pth model for starting, else it will take a pkl or the last pth if exists
    trainer.resume_or_load(resume=True) 
    logger.info("Starting training now...")
    trainer.train()


if __name__ == '__main__':
    args = parse_args()

    main(args.config_file, args.training_classes)
