#!/usr/local/bin/python3
import argparse
import json
import os
import sys
import math
import numpy as np
from PIL import Image
from tools.utils_calib import Calibration
home = os.getenv('HOME')

def parse_args():
    parser = argparse.ArgumentParser(description='Convert kitti dataset to coco format')
    parser.add_argument(
        '--outdir', help="Output directory for JSON files", default=None, type=str)
    parser.add_argument(
        '--datadir', help="Data directory for annotations to be converted",
        default=None, type=str)
    parser.add_argument(
        '--val_file', help="File with the validation imaage names", default='', type=str)
    parser.add_argument(
        '--train_file', help="File with the training image names", default='', type=str)
    parser.add_argument(
        '--bins', help="Bins for viewpoint division", default=16, type=int)
    parser.add_argument(
        '--viewpoint', help="Activates the orientation", default=False, action="store_true")
    parser.add_argument(
        '--vp_res', help="Activates viewpoint correction", default=False, action="store_true")
    parser.add_argument(
        '--only_eval_classes', help="Build the annotations file only with the selected categories", default='0,1,2,3,4,5,6,7,8', type=str)
    parser.add_argument(
        '--rotated_boxes', help="Add rotation_y to the boxes", default=False, action="store_true")
    parser.add_argument(
        '--rotated_dims', help="Change the encoding to fit the object", default=False, action="store_true")
    parser.add_argument(
        '--bv_resolution', help="Birdview resolution to compute not axis-align dimensions", default=0.05, type=float)
    parser.add_argument(
        '--height_enc', help="Adds the encoding for the height and elevation", default=False, action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()
##The expected tree under the --datadir directory is:
##/datadir
##|-- devkit
##|-- testing
##|   |-- calib
##|   |-- image_2
##|   |   |-- <num_image>.txt
##|-- training
##|   |-- calib
##|   |-- image_2
##|   |   |-- <num_image>.txt
##|   |-- label_2
##|   |   |-- <num_image>.txt

def rad2bin(rad, bins):
    bin_dist = np.linspace(-math.pi,math.pi,bins+1) #for each class (bins*n_classes)
    bin_res = (bin_dist[1]-bin_dist[0])/2.
    bin_dist = [bin-bin_res for bin in bin_dist] #Substracting half of the resolution to each bin it obtains one bin for each direction (N,W,S,E)
    for i_bin in range(len(bin_dist)-1):
        if bin_dist[i_bin]<=rad and bin_dist[i_bin+1]>=rad:
            return i_bin

    return 0 #If the angle is above max angle, it won't match so it corresponds to initial bin, initial bin must be from (-pi+bin_res) to (pi-bin_res)

def rad2angle(rad, bins):
    bin_dist = np.linspace(-180,180,bins+1)
    bin_res = (bin_dist[1]-bin_dist[0])/2.
    angle = (-rad)*180/math.pi
    return angle if angle > -bin_res else 180+angle # Half of the resolution (obtaining minimum error) because we need to match angle 0 with 180

def convert_kitti_training(data_dir, out_dir, val_file, train_file, only_eval_classes, bins, viewpoint, vp_res, rdims, height_enc, bvres=0.05, rbox=False):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    subsets=list()
    subsets_files=list()
    if val_file:
        subsets.append('validation')
        subsets_files.append(val_file)
        print("validation file: %s" % val_file)
    if train_file:
        subsets.append('training')
        subsets_files.append(train_file)
        print("training file: %s" % train_file)
    print(subsets,subsets_files)
    only_eval_classes = only_eval_classes.split(',')
    only_eval_classes = [int(cl) for cl in only_eval_classes]
    
    categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
    categories = [categories[idx] for idx in only_eval_classes]
    category_dict = {k:v for v,k in enumerate(categories)}
    images_dir = 'image'
    labels_dir = 'label'
    calibs_dir = 'calib'

    img_id = 0
    ann_id = 0

    strclasses = '' if len(only_eval_classes)==9 else ''.join([el[:3].lower() for el in categories])
    strrot = 'RB' if rbox else '' 
    strrotDim = 'RD' if rdims else '' 
    strH = 'HC' if height_enc else ''
    strVP = 'VP' if viewpoint else '' 
    strVPr = 'r' if vp_res else '' 
    strBINS = str(bins)
    strargs = [strrot,strrotDim,strH,strVP,strVPr,strBINS]
    strargs = ('').join(strargs)

    for sub_set, subset_name in zip(subsets, subsets_files):

        json_name = '{}_annotations_kitti_{}_{}.json'.format(sub_set, strclasses,strargs)
        outfile_name = os.path.join(out_dir, json_name)
        if os.path.exists(outfile_name):
            print('File exists: '+str(outfile_name))
            ans = input("Do you want to overwrite it? (y/n)")
            if ans is 'n':
                # Return always the same file to match with training script
                return os.path.join(out_dir, '{}_annotations_kitti_{}_{}.json'.format('training', strclasses,strargs))

        ann_dir = os.path.join(data_dir, labels_dir)
        im_dir = os.path.join(data_dir, images_dir)
        calib_dir = os.path.join(data_dir, calibs_dir)
        print('Starting %s' % ann_dir)
        print('Starting %s' % im_dir)

        ann_dict = {}
        images = []
        annotations = []

        print("%s" % sub_set)
        print("%s" % subset_name)

        with open(subset_name, "r") as f:
            im_list = f.read().splitlines()
            for filename in im_list:

                complete_name_im = os.path.join(im_dir, filename + '.png')
                complete_name_ann = os.path.join(ann_dir, filename + '.txt')
                complete_name_calib = os.path.join(calib_dir, filename + '.txt')


                image = {}
                image['id'] = img_id
                img_id += 1

                im = Image.open(complete_name_im)
                image['width'], image['height'] = im.size

                image['file_name'] = complete_name_im
                images.append(image)

                velodyne_h = 1.73 # TODO Change to use TF
                calib = Calibration(complete_name_calib)

                pre_objs = np.genfromtxt(complete_name_ann, delimiter=' ',
                    names=['type', 'truncated', 'occluded', 'alpha', 'bbox_xmin', 'bbox_ymin',
                    'bbox_xmax', 'bbox_ymax', 'dimensions_1', 'dimensions_2', 'dimensions_3',
                    'location_1', 'location_2', 'location_3', 'rotation_y'], dtype=None)

                if (pre_objs.ndim < 1):
                    pre_objs = np.array(pre_objs, ndmin=1)

                for obj in pre_objs:
                    o = obj['type']
                    if isinstance(o,(bytes,np.bytes_)):
                        o = o.decode("utf-8")
                    label = category_dict.get(o,8) #Default value just in case
                    if (label != 7) and (label != 8) :
                        # print(obj['type'])
                        ann = {}
                        ann['id'] = ann_id
                        ann_id += 1
                        ann['image_id'] = image['id']
                        ann['category_id'] = label
                        boxes = np.empty((0, 4), dtype=np.float32)
                        ann['bbox'] = [obj['bbox_xmin'], obj['bbox_ymin'], math.fabs(obj['bbox_xmax'] - obj['bbox_xmin']), math.fabs(obj['bbox_ymax'] - obj['bbox_ymin'])]
                        if rbox:
                            rot = rad2angle(obj['rotation_y'],bins)
                            ann['bbox'].append(rot)   
                        # ONLY VALID FOR FRONTAL CAMERA (ONLY_FRONT PARAM)
                        p = calib.project_rect_to_velo(np.array([[obj['location_1'],obj['location_2'],obj['location_3']]]))
                        ann['height'] = [obj['dimensions_1']*255/3.0, ((p[0][2]+velodyne_h)+obj['dimensions_1']*0.5)*255/3.0]#(p[0][2]+velodyne_h)]#Not codificated ground
                        ann['bbox3D'] = [(obj['bbox_xmin']+obj['bbox_xmax'])/2.,(obj['bbox_ymin']+obj['bbox_ymax'])/2., round(obj['dimensions_2']/bvres,3), round(obj['dimensions_3']/bvres,3)]                         # print('ann[bbox]',ann['bbox'])
                        ann['segmentation'] = [[obj['bbox_xmin'], obj['bbox_ymin'], obj['bbox_xmin'], obj['bbox_ymax'], obj['bbox_xmax'], obj['bbox_ymax'], obj['bbox_xmax'], obj['bbox_ymin']]]
                        ann['area'] = math.fabs(obj['bbox_xmax'] - obj['bbox_xmin']) * math.fabs(obj['bbox_ymax'] - obj['bbox_ymin'])
                        ann['iscrowd'] = 0
                        if viewpoint:
                            ann['viewpoint'] = [rad2bin(obj['rotation_y'], bins),obj['rotation_y']] if vp_res else [rad2bin(obj['rotation_y'], bins)]
                        annotations.append(ann)
                        # print(ann)

                if len(images) % 50 == 0:
                    print("Processed %s images, %s annotations" % (
                        len(images), len(annotations)))


        ann_dict['images'] = images
        categories_ = [{"id": category_dict[name], "name": name} for name in
                      category_dict]
        ann_dict['categories'] = categories_
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories_))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))

        # print(ann_dict)
        print("Printed results in: %s" % outfile_name)

        with open(outfile_name, "w") as outfile:
            outfile.write(json.dumps(ann_dict))
    return outfile_name


def convert_kitti_testing(data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
    images_dir =  'image_2'
    json_name = 'annotations_kitti_%s.json'

    category_dict = {'Car' : 0,
                    'Van' : 1,
                    'Truck' : 2,
                    'Pedestrian' : 3,
                    'Person_sitting' : 4,
                    'Cyclist' : 5,
                    'Tram' : 6,
                    'Misc' : 7,
                    'DontCare' : 8
                   }
    img_id = 0
    ann_id = 0

    data_set = 'testing'
    im_dir = os.path.join(data_dir, data_set, images_dir)
    print('Starting %s' % im_dir)

    ann_dict = {}
    images = []
    annotations = []

    for root, _, files in os.walk(im_dir):
        print('Root %s' % root)
        for filename in files:

            complete_name_im = os.path.join(im_dir, filename)
            filename, file_extension = os.path.splitext(filename)

            image = {}
            image['id'] = img_id
            img_id += 1

            im = Image.open(complete_name_im)
            image['width'], image['height'] = im.size

            image['file_name'] = complete_name_im
            images.append(image)

            if len(images) % 50 == 0:
                print("Processed %s images, %s annotations" % (
                    len(images), len(annotations)))


    ann_dict['images'] = images
    categories = [{"id": category_dict[name], "name": name} for name in
                  category_dict]
    ann_dict['categories'] = categories
    ann_dict['annotations'] = annotations
    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))

    outfile_name = os.path.join(out_dir, json_name % data_set)
    print("Printed results in: %s" % outfile_name)

    with open(outfile_name, 'wb') as outfile:
        outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()

    convert_kitti_training(args.datadir, args.outdir, args.val_file, args.train_file, args.only_eval_classes, args.bins, args.viewpoint, args.vp_res, args.rotated_dims, args.height_enc, args.bv_resolution, args.rotated_boxes)
    #convert_kitti_testing(args.datadir, args.outdir)
